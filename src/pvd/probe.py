"""Frozen-embedding linear probe: CV tuning, save/load, training (M4, §9.8).

The probe is a logistic regression on cached embeddings. It retrains in seconds,
which is what makes the review loop responsive. Cross-validation picks the
regularization strength; the review loop reuses the last chosen ``C`` for speed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from .config import Config
from .evaluate import evaluate_if_test_set, render_report, write_report
from .features import feature_recipe, get_features, load_feature_meta, package_versions
from .labels import effective_labels, label_snapshot_sha
from .scan import load_manifest

log = logging.getLogger(__name__)

MIN_TRAIN_PER_CLASS = 5


@dataclass
class ProbeBundle:
    """Everything needed to reproduce and validate a trained probe (§8.4)."""

    kind: str
    classes: list[str]
    backbone_id: str
    feature_dim: int
    estimator: object
    feature_meta: dict = field(default_factory=dict)
    #: Full cache key (backbone + tiles/extra/background suffixes) and the recipe
    #: that produced it, so predict/eval rebuild exactly these features (G8).
    feature_key: str = ""
    feature_recipe: dict = field(default_factory=dict)
    best_C: float | None = None
    cv_balanced_accuracy: float | None = None
    test_metrics: dict | None = None
    n_train: int = 0
    label_snapshot_sha: str = ""
    config: dict = field(default_factory=dict)
    created: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    versions: dict = field(default_factory=dict)


def make_estimator(C: float, config: Config) -> LogisticRegression:
    """The probe's estimator: balanced class weights so imbalance can't bias it (G27)."""
    return LogisticRegression(
        C=C,
        class_weight=config.probe.class_weight,
        max_iter=config.probe.max_iter,
    )


def _pick_best_C(search: GridSearchCV) -> float:
    """Choose the strongest regularization that still gives the best CV score.

    When several ``C`` values tie on balanced accuracy (common once classes are
    separable), GridSearchCV returns the first/smallest. That heavily regularizes
    and squashes predicted probabilities toward 0.5, which makes any confidence
    threshold useless. We prefer the largest tied ``C`` for sharper, still-honest
    probabilities.
    """
    scores = np.asarray(search.cv_results_["mean_test_score"], dtype=float)
    candidates = np.asarray([float(c) for c in search.cv_results_["param_C"]], dtype=float)
    tied = candidates[scores >= scores.max() - 1e-9]
    return float(tied.max())


def train_probe(
    X: np.ndarray,
    y: np.ndarray,
    config: Config,
    *,
    classes: list[str],
    backbone_id: str,
    feature_meta: dict | None = None,
    test_metrics: dict | None = None,
    label_snapshot: str = "",
    recipe: dict | None = None,
) -> ProbeBundle:
    """Grid-search ``C`` by stratified CV and fit the final estimator (G13)."""
    if len(X) != len(y):
        raise ValueError("X and y must have the same length")
    counts = np.bincount(y, minlength=len(classes))
    if len(counts) != len(classes) or (counts == 0).any():
        # predict_proba would return fewer columns than there are classes and
        # every caller indexing columns by class position would be misaligned.
        missing = [name for name, n in zip(classes, counts) if n == 0]
        raise ValueError(f"Every class needs labeled examples to train a probe; missing: {missing}")
    if counts.min() < 2:
        raise ValueError("Each class needs at least two labeled examples for CV")

    folds = min(config.probe.cv_folds, int(counts.min()))
    search = GridSearchCV(
        make_estimator(1.0, config),
        {"C": config.probe.C_grid},
        scoring="balanced_accuracy",
        cv=StratifiedKFold(n_splits=max(2, folds), shuffle=True, random_state=config.seed),
        n_jobs=1,
    )
    search.fit(X, y)
    best_C = _pick_best_C(search)
    estimator = make_estimator(best_C, config).fit(X, y)
    log.info("Probe: best C=%.3g, CV balanced accuracy=%.3f", best_C, search.best_score_)

    return ProbeBundle(
        kind="probe",
        classes=list(classes),
        backbone_id=backbone_id,
        feature_dim=int(X.shape[1]),
        estimator=estimator,
        feature_meta=feature_meta or {},
        feature_key=(recipe or {}).get("feature_key", ""),
        feature_recipe=recipe or {},
        best_C=best_C,
        cv_balanced_accuracy=float(search.best_score_),
        test_metrics=test_metrics,
        n_train=int(len(y)),
        label_snapshot_sha=label_snapshot,
        config=config.to_dict(),
        versions=package_versions(),
    )


def save(bundle: ProbeBundle, path: Path) -> Path:
    """Write a bundle with joblib (a pickle: only load bundles you trust)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    log.info("Saved model bundle: %s", path)
    return path


def run_train(config: Config) -> Path:
    """Train and save a probe from the current effective labels."""
    manifest = load_manifest(config)
    labels = effective_labels(config, manifest)
    if labels.empty:
        raise ValueError(
            "No labeled training data. Add examples to data/seed/<class>/ or label "
            "images in `pvd review` first."
        )

    class_to_index = {name: i for i, name in enumerate(config.classes)}
    counts = labels["label"].value_counts()
    small = [name for name in config.classes if counts.get(name, 0) < MIN_TRAIN_PER_CLASS]
    if small:
        raise ValueError(
            f"Not enough labeled examples for {small} (need >= {MIN_TRAIN_PER_CLASS} "
            "each). Until then use `pvd zeroshot`, or add seed images / label more "
            "in `pvd review` (which uses zero-shot until every class has enough)."
        )

    shas = labels["sha1"].tolist()
    paths = [str(config.resolve(p)) for p in labels["path"].tolist()]
    X, valid = get_features(config, shas, paths, return_mask=True)
    if not valid.all():
        log.warning("Dropping %d labeled image(s) without embeddings.", int((~valid).sum()))
        X, labels = X[valid], labels[valid].reset_index(drop=True)
    y = np.array([class_to_index[label] for label in labels["label"]], dtype=int)

    recipe = feature_recipe(config.backbone.id, config)
    bundle = train_probe(
        X,
        y,
        config,
        classes=list(config.classes),
        backbone_id=config.backbone.id,
        feature_meta=load_feature_meta(config),
        label_snapshot=label_snapshot_sha(labels),
        recipe=recipe,
    )

    result = evaluate_if_test_set(config, bundle, manifest)
    if result is not None:
        bundle.test_metrics = {
            "accuracy": result.accuracy,
            "balanced_accuracy": result.balanced_accuracy,
            "auc": result.auc,
            "per_tag": result.per_tag,
            "recommended_threshold": result.recommended_threshold,
        }

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = save(bundle, config.models_dir / f"probe_{stamp}.joblib")

    class_counts = ", ".join(f"{name}={int((y == i).sum())}" for i, name in enumerate(config.classes))
    lines = [
        f"- Training examples: **{bundle.n_train}** ({class_counts}; class_weight=balanced)",
        f"- Best C: **{bundle.best_C}**",
        f"- CV balanced accuracy: **{bundle.cv_balanced_accuracy:.3f}**",
        f"- Label snapshot: `{bundle.label_snapshot_sha[:12]}`",
        f"- Bundle: `{path.name}`",
        "",
    ]
    if result is not None:
        lines.insert(2, f"- Test balanced accuracy: **{result.balanced_accuracy:.3f}**")
        report = write_report(config, "train", render_report("probe", result, lines))
    else:
        report = write_report(config, "train", "# Training report\n\n" + "\n".join(lines))

    log.info("Training report: %s", report)
    return path
