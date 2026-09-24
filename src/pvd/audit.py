"""Audit tools: label errors, near-duplicate leakage and shortcut learning (G6, G21, G28).

These are the "am I fooling myself?" checks. They all operate on cached
embeddings and labels, so they run in seconds.

The label-error check uses **out-of-fold** predictions (``cross_val_predict``).
In-sample predictions would just echo the labels we gave the model and hide the
mistakes we are trying to find.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from .config import Config
from .evaluate import write_report
from .features import get_features
from .labels import effective_labels
from .probe import make_estimator
from .scan import load_manifest

log = logging.getLogger(__name__)

SUSPECT_MAX_OWN_PROB = 0.20
NEAR_DUPLICATE_COSINE = 0.97
SHORTCUT_WARN_ACCURACY = 0.95


def _labeled_with_features(config: Config, manifest: pd.DataFrame):
    """Return ``(X, y, labels, manifest_rows)`` aligned for labeled examples."""
    labels = effective_labels(config, manifest)
    if labels.empty:
        return None
    positions = {sha: i for i, sha in enumerate(manifest["sha1"])}
    labels = labels[labels["sha1"].isin(positions)].reset_index(drop=True)
    if labels.empty:
        return None
    shas = labels["sha1"].tolist()
    paths = [str(config.resolve(p)) for p in labels["path"].tolist()]
    X = get_features(config, shas, paths)
    class_to_index = {name: i for i, name in enumerate(config.classes)}
    y = np.array([class_to_index[label] for label in labels["label"]], dtype=int)
    attribute_rows = manifest.iloc[[positions[sha] for sha in shas]].reset_index(drop=True)
    return X, y, labels, attribute_rows


def label_error_suspects(config: Config, manifest: pd.DataFrame | None = None) -> list[dict]:
    """Items whose own label the out-of-fold model finds improbable."""
    manifest = manifest if manifest is not None else load_manifest(config)
    prepared = _labeled_with_features(config, manifest)
    if prepared is None:
        return []
    X, y, labels, _ = prepared
    counts = np.bincount(y, minlength=len(config.classes))
    if counts.min() < 2 or X.shape[0] < 4:
        return []
    folds = min(config.probe.cv_folds, int(counts.min()))
    # A sharp (weakly-regularized) model gives confident out-of-fold probabilities,
    # which is what makes a low own-label probability meaningful.
    estimator = make_estimator(max(config.probe.C_grid), config)
    probs = cross_val_predict(
        estimator,
        X,
        y,
        cv=StratifiedKFold(n_splits=max(2, folds), shuffle=True, random_state=config.seed),
        method="predict_proba",
    )
    own_prob = probs[np.arange(len(y)), y]
    order = np.argsort(own_prob)
    suspects = []
    for i in order:
        if own_prob[i] >= SUSPECT_MAX_OWN_PROB:
            break
        suspects.append(
            {
                "sha1": labels.iloc[i]["sha1"],
                "path": labels.iloc[i]["path"],
                "label": labels.iloc[i]["label"],
                "oof_prob": float(own_prob[i]),
            }
        )
    return suspects


def near_duplicate_pairs(
    config: Config, manifest: pd.DataFrame | None = None, limit: int = 50
) -> list[dict]:
    """Test images that are near-identical to a training image (leakage, G6)."""
    manifest = manifest if manifest is not None else load_manifest(config)
    train = manifest[(manifest["split"].isin(["seed", "pool"])) & (manifest["status"] == "ok")]
    test = manifest[(manifest["split"] == "test") & (manifest["status"] == "ok")]
    if train.empty or test.empty:
        return []
    train_shas = train["sha1"].tolist()
    test_shas = test["sha1"].tolist()
    X_train = get_features(config, train_shas, [str(config.resolve(p)) for p in train["path"]])
    X_test = get_features(config, test_shas, [str(config.resolve(p)) for p in test["path"]])

    from .labels import LabelStore

    # Only images that actually carry a training label matter for leakage.
    store = LabelStore(config.labels_file)
    overrides = store.current()
    labels = effective_labels(config, manifest)
    train_labeled = set(labels["sha1"]) | {
        sha for sha in train_shas if sha in overrides
    }
    keep = [i for i, sha in enumerate(train_shas) if sha in train_labeled]
    if not keep:
        return []
    X_train = X_train[keep]

    similarity = X_test @ X_train.T
    best = similarity.max(axis=1)
    order = np.argsort(-best)
    pairs = []
    for i in order:
        if best[i] < NEAR_DUPLICATE_COSINE:
            break
        j = int(similarity[i].argmax())
        pairs.append(
            {
                "test_path": test.iloc[i]["path"],
                "train_path": train.iloc[keep[j]]["path"],
                "similarity": float(best[i]),
            }
        )
        if len(pairs) >= limit:
            break
    return pairs


def shortcut_analysis(config: Config, manifest: pd.DataFrame | None = None) -> dict:
    """Check whether nuisance attributes predict the class (or the class them).

    Two questions per attribute:
    1. How well is the *class* predictable from the attribute value alone? (bad)
    2. How well is the *attribute* predictable from the embeddings? (leak signal)
    """
    manifest = manifest if manifest is not None else load_manifest(config)
    prepared = _labeled_with_features(config, manifest)
    if prepared is None:
        return {}
    X, y, labels, attribute_rows = prepared

    attributes = {
        "format": attribute_rows["format"].fillna("?").astype(str).tolist(),
        "has_alpha": [str(bool(v)) for v in attribute_rows["has_alpha"].fillna(False).tolist()],
        "source_dir": attribute_rows["source_dir"].fillna("?").astype(str).tolist(),
    }
    results: dict[str, dict] = {}
    for name, values in attributes.items():
        values_array = np.array(values)
        buckets = sorted(set(values))
        # (1) Predict class from the attribute value (majority-per-bucket).
        correct = 0
        for bucket in buckets:
            mask = values_array == bucket
            if mask.sum() == 0:
                continue
            counts = np.bincount(y[mask], minlength=len(config.classes))
            correct += int(counts.max())
        class_from_attr = correct / len(y) if len(y) else 0.0
        # (2) Predict the attribute from embeddings (only if 2+ buckets).
        attr_from_embedding = None
        if len(buckets) >= 2 and min((values_array == b).sum() for b in buckets) >= 4:
            try:
                auc_scores = cross_val_predict(
                    LogisticRegression(max_iter=config.probe.max_iter),
                    X,
                    values_array,
                    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=config.seed),
                    method="predict_proba",
                )
                # Binary accuracy from the class column order of the estimator.
                classes = np.unique(values_array)
                predicted = classes[auc_scores.argmax(axis=1)]
                attr_from_embedding = float((predicted == values_array).mean())
            except Exception as exc:  # noqa: BLE001
                log.debug("shortcut probe for %s failed: %s", name, exc)
        results[name] = {
            "buckets": {b: int((values_array == b).sum()) for b in buckets},
            "class_from_attribute_accuracy": class_from_attr,
            "attribute_from_embedding_accuracy": attr_from_embedding,
            "warn": class_from_attr >= SHORTCUT_WARN_ACCURACY,
        }
    return results


def render_audit_report(suspects: list[dict], pairs: list[dict], shortcuts: dict) -> str:
    lines = ["# Audit report", ""]
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    lines.append("")

    lines.append(f"## Label-error suspects ({len(suspects)})")
    lines.append("")
    lines.append("Out-of-fold probability for the *given* label below 0.20, worst first.")
    lines.append("")
    if suspects:
        lines.append("| path | label | oof prob |")
        lines.append("|---|---|---|")
        for s in suspects:
            lines.append(f"| `{s['path']}` | {s['label']} | {s['oof_prob']:.3f} |")
    else:
        lines.append("None. (Good, or not enough labels yet.)")
    lines.append("")

    lines.append(f"## Near-duplicate leakage ({len(pairs)})")
    lines.append("")
    lines.append(f"Test/train pairs with cosine similarity > {NEAR_DUPLICATE_COSINE}.")
    lines.append("")
    if pairs:
        lines.append("| test | train | similarity |")
        lines.append("|---|---|---|")
        for p in pairs:
            lines.append(f"| `{p['test_path']}` | `{p['train_path']}` | {p['similarity']:.3f} |")
    else:
        lines.append("None.")
    lines.append("")

    lines.append("## Shortcut analysis")
    lines.append("")
    lines.append("`class_from_attribute` near 1.0 means the class is almost readable")
    lines.append("from a nuisance attribute alone; a high `attribute_from_embedding` means")
    lines.append("the embedding encodes the nuisance attribute (a shortcut risk).")
    lines.append("")
    lines.append("| attribute | buckets | class_from_attribute | attribute_from_embedding | warn |")
    lines.append("|---|---|---|---|---|")
    for name, info in shortcuts.items():
        buckets = ", ".join(f"{b}={n}" for b, n in info["buckets"].items())
        from_attr = f"{info['class_from_attribute_accuracy'] * 100:.1f}%"
        from_embed = (
            f"{info['attribute_from_embedding_accuracy'] * 100:.1f}%"
            if info["attribute_from_embedding_accuracy"] is not None
            else "n/a"
        )
        lines.append(
            f"| {name} | {buckets} | {from_attr} | {from_embed} | {'YES' if info['warn'] else 'no'} |"
        )
    lines.append("")
    warnings = [name for name, info in shortcuts.items() if info["warn"]]
    if warnings:
        lines.append(
            f"> **Warning:** class is almost perfectly predictable from {warnings}. "
            "Balance the training data across these attributes before trusting the model."
        )
    return "\n".join(lines)


def run_audit(config: Config) -> Path:
    """Run all audit checks and write a Markdown report."""
    manifest = load_manifest(config)
    suspects = label_error_suspects(config, manifest)
    pairs = near_duplicate_pairs(config, manifest)
    shortcuts = shortcut_analysis(config, manifest)
    report = write_report(config, "audit", render_audit_report(suspects, pairs, shortcuts))
    log.info(
        "Audit: %d suspect labels, %d near-duplicate pairs, %d shortcut warnings",
        len(suspects),
        len(pairs),
        sum(1 for info in shortcuts.values() if info["warn"]),
    )
    return report
