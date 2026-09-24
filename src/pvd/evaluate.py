"""Honest evaluation: metrics, per-tag and bias breakdowns, reports (G11-G15, G28).

All reported numbers come from data the model never trained on: the held-out
``data/test`` split. The review UI deliberately shows hard examples, so its
"wrong on N" counter is only a hint; these reports are the real progress meter.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from .config import Config
from .scan import load_manifest

log = logging.getLogger(__name__)

# Long-side buckets, half-open so every size lands in exactly one bucket.
RESOLUTION_BUCKETS = [
    (0, 512, "<512"),
    (512, 1024, "512-1023"),
    (1024, 2048, "1024-2047"),
    (2048, 10**9, ">=2048"),
]


# --------------------------------------------------------------------------- #
# Test-set assembly
# --------------------------------------------------------------------------- #
@dataclass
class TestSet:
    """The held-out split, aligned row by row: frame, sha1s, paths, labels, tags."""

    frame: pd.DataFrame
    shas: list[str]
    paths: list[str]
    y_true: np.ndarray
    tags: list[str | None]

    def subset(self, mask: np.ndarray) -> "TestSet":
        """Keep only rows where ``mask`` is True (e.g. images that embedded)."""
        mask = np.asarray(mask, dtype=bool)
        index = np.flatnonzero(mask)
        return TestSet(
            frame=self.frame.iloc[index].reset_index(drop=True),
            shas=[self.shas[i] for i in index],
            paths=[self.paths[i] for i in index],
            y_true=self.y_true[index],
            tags=[self.tags[i] for i in index],
        )


def resolve_test_set(config: Config, manifest: pd.DataFrame | None = None) -> TestSet:
    """Return the test split with integer labels, or raise a clear error.

    Pass an already-loaded ``manifest`` to avoid re-reading it from disk (the
    review app evaluates after every batch).
    """
    manifest = manifest if manifest is not None else load_manifest(config)
    test = manifest[(manifest["split"] == "test") & (manifest["status"] == "ok")].copy()
    if test.empty:
        raise ValueError(
            f"No usable test images under {config.test_dir}. "
            "Add images to data/test/<class>/[<tag>/] to get honest metrics."
        )
    class_to_index = {name: i for i, name in enumerate(config.classes)}
    unknown = set(test["folder_label"]) - set(class_to_index)
    if unknown:
        raise ValueError(f"Test images under unknown classes: {sorted(unknown)}")
    test = test[test["folder_label"].isin(class_to_index)].reset_index(drop=True)
    return TestSet(
        frame=test,
        shas=test["sha1"].tolist(),
        paths=[str(config.resolve(p)) for p in test["path"].tolist()],
        y_true=np.array([class_to_index[label] for label in test["folder_label"]], dtype=int),
        tags=test["tag"].tolist(),
    )


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
@dataclass
class EvalResult:
    classes: list[str]
    accuracy: float
    balanced_accuracy: float
    per_class: dict[str, dict[str, float]]
    confusion: list[list[int]]
    auc: float | None
    per_tag: dict[str, dict[str, float]] = field(default_factory=dict)
    bias: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)
    coverage: list[tuple[float, float, float]] = field(default_factory=list)
    recommended_threshold: float | None = None
    worst_errors: list[dict] = field(default_factory=list)


def _basic_metrics(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(classes))), zero_division=0
    )
    per_class = {
        classes[i]: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(len(classes))
    }
    # Balanced accuracy is meaningless (and warns) when a subgroup has one class.
    balanced = (
        float(balanced_accuracy_score(y_true, y_pred))
        if len(np.unique(y_true)) >= 2
        else float("nan")
    )
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": balanced,
        "per_class": per_class,
    }


def _auc(y_true: np.ndarray, probs: np.ndarray, n_classes: int) -> float | None:
    try:
        if n_classes == 2:
            return float(roc_auc_score(y_true, probs[:, 1]))
        return float(
            roc_auc_score(
                y_true, probs, multi_class="ovr", average="macro", labels=list(range(n_classes))
            )
        )
    except Exception:  # noqa: BLE001 - too few samples per class, etc.
        return None


def coverage_accuracy(
    probs: np.ndarray, y_pred: np.ndarray, y_true: np.ndarray, target: float = 0.98
) -> tuple[list[tuple[float, float, float]], float | None]:
    """Coverage/accuracy table for the 'unsure' threshold and a recommendation.

    For each threshold, images whose max probability is below it are left
    undecided; the rest are 'covered'. We recommend the lowest threshold whose
    decided-set accuracy meets ``target`` (and that decides something).
    """
    confidence = probs.max(axis=1)
    rows: list[tuple[float, float, float]] = []
    recommended: float | None = None
    for threshold in np.round(np.arange(0.50, 0.995, 0.01), 2):
        decided = confidence >= threshold
        coverage = float(decided.mean())
        accuracy = float((y_pred[decided] == y_true[decided]).mean()) if decided.any() else float("nan")
        rows.append((float(threshold), coverage, accuracy))
        if recommended is None and decided.any() and accuracy >= target:
            recommended = float(threshold)
    return rows, recommended


def evaluate_arrays(
    y_true: np.ndarray,
    probs: np.ndarray,
    classes: list[str],
    frame: pd.DataFrame | None = None,
    tags: list[str | None] | None = None,
) -> EvalResult:
    """Compute the full metric set from probabilities (section 9.10)."""
    y_pred = probs.argmax(axis=1)
    base = _basic_metrics(y_true, y_pred, classes)
    result = EvalResult(
        classes=classes,
        accuracy=base["accuracy"],
        balanced_accuracy=base["balanced_accuracy"],
        per_class=base["per_class"],
        confusion=confusion_matrix(y_true, y_pred, labels=list(range(len(classes)))).tolist(),
        auc=_auc(y_true, probs, len(classes)),
    )

    coverage, recommended = coverage_accuracy(probs, y_pred, y_true)
    result.coverage = coverage
    result.recommended_threshold = recommended

    if tags is not None:
        for tag in sorted({t for t in tags if t}):
            mask = np.array([t == tag for t in tags])
            if mask.sum() == 0:
                continue
            m = _basic_metrics(y_true[mask], y_pred[mask], classes)
            result.per_tag[tag] = {"n": m["n"], "accuracy": m["accuracy"], "balanced_accuracy": m["balanced_accuracy"]}

    if frame is not None:
        result.bias = bias_breakdowns(frame, y_true, y_pred, classes)
        result.worst_errors = worst_errors(frame, y_true, y_pred, probs, classes)

    return result


def _resolution_bucket(width: int, height: int) -> str:
    long_side = max(width, height)
    for low, high, name in RESOLUTION_BUCKETS:
        if low <= long_side < high:
            return name
    return ">2048"


def bias_breakdowns(
    frame: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]
) -> dict[str, dict[str, dict[str, float]]]:
    """Accuracy sliced by nuisance attributes that must not drive the label (G28)."""
    attributes: dict[str, list] = {
        "format": frame["format"].fillna("?").tolist(),
        "has_alpha": [str(bool(v)) for v in frame["has_alpha"].fillna(False).tolist()],
        "resolution": [
            _resolution_bucket(int(w), int(h))
            for w, h in zip(frame["width"].fillna(0), frame["height"].fillna(0))
        ],
        "source_dir": frame["source_dir"].fillna("?").tolist(),
    }
    breakdown: dict[str, dict[str, dict[str, float]]] = {}
    for name, values in attributes.items():
        groups: dict[str, dict[str, float]] = {}
        for bucket in sorted(set(values)):
            mask = np.array([v == bucket for v in values])
            if mask.sum() == 0:
                continue
            groups[str(bucket)] = {
                "n": int(mask.sum()),
                "accuracy": float((y_pred[mask] == y_true[mask]).mean()),
            }
        breakdown[name] = groups
    return breakdown


def worst_errors(
    frame: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    classes: list[str],
    limit: int = 12,
) -> list[dict]:
    confidence = probs.max(axis=1)
    wrong = np.where(y_true != y_pred)[0]
    wrong = wrong[np.argsort(-confidence[wrong])][:limit]
    return [
        {
            "path": frame.iloc[i]["path"],
            "true": classes[y_true[i]],
            "pred": classes[y_pred[i]],
            "confidence": float(confidence[i]),
        }
        for i in wrong
    ]


# --------------------------------------------------------------------------- #
# Render and write
# --------------------------------------------------------------------------- #
def _pct(value: float) -> str:
    return f"{value * 100:.2f}%" if value == value else "n/a"


def render_report(name: str, result: EvalResult, extra_lines: list[str] | None = None) -> str:
    """Render an :class:`EvalResult` as the standard Markdown evaluation report."""
    lines: list[str] = []
    lines.append(f"# Evaluation report: {name}")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    lines.append("")
    lines.append(f"- Accuracy: **{_pct(result.accuracy)}**")
    lines.append(f"- Balanced accuracy: **{_pct(result.balanced_accuracy)}**")
    lines.append(f"- ROC-AUC: **{_pct(result.auc) if result.auc is not None else 'n/a'}**")
    lines.append("")

    lines.append("## Per-class")
    lines.append("")
    lines.append("| class | precision | recall | f1 | support |")
    lines.append("|---|---|---|---|---|")
    for cls, m in result.per_class.items():
        lines.append(
            f"| {cls} | {_pct(m['precision'])} | {_pct(m['recall'])} | {_pct(m['f1'])} | {m['support']} |"
        )
    lines.append("")

    lines.append("## Confusion matrix (rows=true, cols=pred)")
    lines.append("")
    header = "| | " + " | ".join(result.classes) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(result.classes) + 1))
    for i, cls in enumerate(result.classes):
        row = " | ".join(str(v) for v in result.confusion[i])
        lines.append(f"| {cls} | {row} |")
    lines.append("")

    if result.per_tag:
        lines.append("## Per-tag")
        lines.append("")
        lines.append("| tag | n | accuracy | balanced accuracy |")
        lines.append("|---|---|---|---|")
        for tag, m in result.per_tag.items():
            lines.append(f"| {tag} | {m['n']} | {_pct(m['accuracy'])} | {_pct(m['balanced_accuracy'])} |")
        lines.append("")

    if result.bias:
        lines.append("## Bias breakdowns")
        lines.append("")
        for attr, groups in result.bias.items():
            lines.append(f"### {attr}")
            lines.append("")
            lines.append("| bucket | n | accuracy |")
            lines.append("|---|---|---|")
            for bucket, m in groups.items():
                lines.append(f"| {bucket} | {m['n']} | {_pct(m['accuracy'])} |")
            lines.append("")

    lines.append("## Coverage vs accuracy (unsure threshold)")
    lines.append("")
    lines.append("| threshold | coverage | accuracy on decided |")
    lines.append("|---|---|---|")
    for threshold, coverage, accuracy in result.coverage[::5]:
        lines.append(f"| {threshold:.2f} | {_pct(coverage)} | {_pct(accuracy)} |")
    lines.append("")
    if result.recommended_threshold is not None:
        lines.append(
            f"**Recommended unsure_threshold: {result.recommended_threshold:.2f}** "
            "(lowest threshold reaching 98% accuracy on decided images)."
        )
    else:
        lines.append("No threshold reached 98% accuracy on decided images.")
    lines.append("")

    if result.worst_errors:
        lines.append("## Worst test errors")
        lines.append("")
        lines.append("| path | true | predicted | confidence |")
        lines.append("|---|---|---|---|")
        for error in result.worst_errors:
            lines.append(
                f"| `{error['path']}` | {error['true']} | {error['pred']} | {_pct(error['confidence'])} |"
            )
        lines.append("")

    if extra_lines:
        lines.extend(extra_lines)
    return "\n".join(lines)


def write_report(config: Config, kind: str, markdown: str) -> Path:
    """Save a Markdown report as ``reports/<UTC timestamp>_<kind>.md``."""
    config.reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = config.reports_dir / f"{stamp}_{kind}.md"
    path.write_text(markdown, encoding="utf-8")
    log.info("Wrote report: %s", path)
    return path


# --------------------------------------------------------------------------- #
# Trained model evaluation
# --------------------------------------------------------------------------- #
def evaluate_bundle(config: Config, bundle, test: TestSet | None = None) -> EvalResult:
    """Score a probe or fine-tuned bundle on the test set, in config class order.

    Images that cannot be embedded/decoded are dropped (and counted) rather
    than scored as zero vectors; probability columns are matched to the config
    classes by name, so a bundle trained before the classes were reordered is
    still scored correctly.
    """
    from .models import align_probs, predict_proba_paths

    test = test if test is not None else resolve_test_set(config)
    probs, valid = predict_proba_paths(bundle, test.paths, config, shas=test.shas)
    if not valid.all():
        log.warning("Evaluating without %d test image(s) that could not be embedded.", int((~valid).sum()))
        test, probs = test.subset(valid), probs[valid]
    if not len(test.shas):
        raise ValueError("No test image could be embedded; nothing to evaluate.")
    probs = align_probs(probs, list(bundle.classes), list(config.classes))
    return evaluate_arrays(test.y_true, probs, list(config.classes), test.frame, test.tags)


def run_eval(config: Config, model: str | Path | None = None) -> Path:
    """Evaluate the newest (or given) bundle on the held-out test set."""
    from .models import load_any, load_latest_any

    path = Path(model) if model else load_latest_any(config)
    bundle = load_any(path)
    result = evaluate_bundle(config, bundle)
    extra = [
        f"Model: `{path.name}`",
        f"Kind: `{getattr(bundle, 'kind', 'probe')}`",
        f"Trained on {bundle.n_train} examples (snapshot `{bundle.label_snapshot_sha[:12]}`)",
        "",
    ]
    report = write_report(config, "eval", render_report(path.stem, result, extra))
    log.info("Test balanced accuracy: %.3f", result.balanced_accuracy)
    return report


# --------------------------------------------------------------------------- #
# Specialist baseline (deepghs)
# --------------------------------------------------------------------------- #
def run_specialist_baseline(config: Config, name: str, score_fn) -> Path:
    """Evaluate a single-score 'P(real)' model on the test set (G15)."""
    test = resolve_test_set(config)
    if list(config.classes) != ["photo", "drawing"]:
        raise ValueError(
            "Specialist baselines assume classes == [photo, drawing]; "
            f"got {config.classes}."
        )
    probs = np.zeros((len(test.shas), 2), dtype=np.float64)
    failures = 0
    for i, path in enumerate(test.paths):
        try:
            p_real = float(score_fn(path))
        except Exception as exc:  # noqa: BLE001
            failures += 1
            log.error("Baseline failed on %s: %s", path, exc)
            p_real = 0.5
        probs[i] = [p_real, 1.0 - p_real]
    if failures:
        log.warning("%d test images failed for baseline %s", failures, name)

    result = evaluate_arrays(test.y_true, probs, list(config.classes), test.frame, test.tags)
    extra = [f"Model: `{name}` (P(real) mapped to `photo`)", ""]
    if failures:
        extra.append(f"> {failures} images could not be scored and defaulted to 0.5")
    report = write_report(config, f"baseline_{name}", render_report(name, result, extra))
    log.info("Balanced accuracy: %.3f", result.balanced_accuracy)
    return report
