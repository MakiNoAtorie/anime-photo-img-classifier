"""Unified model loading and prediction across probe and fine-tuned bundles.

The probe predicts from cached embeddings; the fine-tuned model predicts from
raw images. Callers (``predict``, ``eval``) should not care which they hold, so
this module hides the difference behind :func:`predict_proba_paths`.

Two safety rules live here:

- A probe is always fed features built with **its own** recipe (backbone, tiles,
  extra block, alpha background), taken from the bundle rather than from the
  current ``config.yaml`` (G8, NF4). Editing the config after training can no
  longer silently change what the model sees.
- Probabilities are returned in the bundle's class order; :func:`align_probs`
  reorders them by *name* to whatever order the caller needs.

Security note: bundles are pickles (joblib), and unpickling runs code. Only load
bundles you created or trust. Auto-discovery reads only this project's own
``probe_*``/``finetune_*`` files, newest first, and stops at the first match.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np

from .config import Config
from .features import feature_key, get_features
from .imageio import sha1_file

log = logging.getLogger(__name__)

KNOWN_KINDS = {"probe", "finetune"}


def load_any(path: Path):
    """Load either a probe or a fine-tuned bundle."""
    bundle = joblib.load(path)
    kind = getattr(bundle, "kind", "probe")
    if kind not in KNOWN_KINDS:
        raise ValueError(f"Unknown model bundle kind {kind!r} in {path}")
    return bundle


def newest_bundle_path(config: Config, kinds: tuple[str, ...] = ("probe", "finetune")) -> Path:
    """Newest ``models/<kind>_*.joblib`` by modification time, without unpickling.

    Choosing by file name avoids loading (and so executing) every pickle in the
    folder just to find the right one.
    """
    candidates = [
        path
        for kind in kinds
        for path in config.models_dir.glob(f"{kind}_*.joblib")
    ]
    if not candidates:
        wanted = " or ".join(kinds)
        raise FileNotFoundError(
            f"No {wanted} model found in {config.models_dir}. Run `pvd train` first."
        )
    return max(candidates, key=lambda p: (p.stat().st_mtime, p.name))


def config_for_bundle(config: Config, bundle) -> Config:
    """A config copy whose feature settings match the bundle's training recipe."""
    recipe = getattr(bundle, "feature_recipe", None) or {}
    if not recipe:
        return config
    cfg = config.model_copy(deep=True)
    cfg.backbone.id = recipe.get("backbone", cfg.backbone.id)
    if "tiles" in recipe:
        cfg.backbone.tiles = type(cfg.backbone.tiles).model_validate(recipe["tiles"])
    cfg.backbone.extra_features.deepghs_real = bool(recipe.get("deepghs_real", False))
    if "alpha_background" in recipe:
        cfg.image.alpha_background = tuple(recipe["alpha_background"])
    return cfg


def align_probs(probs: np.ndarray, from_classes: list[str], to_classes: list[str]) -> np.ndarray:
    """Reorder probability columns from ``from_classes`` to ``to_classes`` by name.

    ``eval`` builds ``y_true`` in config order; a bundle trained before the
    classes were reordered would otherwise be scored against the wrong columns.
    """
    if list(from_classes) == list(to_classes):
        return probs
    if set(from_classes) != set(to_classes):
        raise ValueError(f"Model classes {from_classes} != config classes {to_classes}.")
    order = [list(from_classes).index(name) for name in to_classes]
    return probs[:, order]


def predict_proba_paths(
    bundle,
    paths: list[str],
    config: Config,
    shas: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(probs, valid)`` for image paths, in the bundle's class order.

    ``probs`` has one row per path; rows where ``valid`` is False could not be
    embedded or decoded and hold NaN, so a caller cannot use them by accident.
    """
    kind = getattr(bundle, "kind", "probe")
    if kind == "finetune":
        from .finetune import finetune_predict_proba

        return finetune_predict_proba(bundle, paths, config)

    cfg = config_for_bundle(config, bundle)
    expected_key = getattr(bundle, "feature_key", "") or ""
    actual_key = feature_key(bundle.backbone_id, cfg)
    if expected_key and expected_key != actual_key:
        raise ValueError(
            f"Model expects features {expected_key!r} but would get {actual_key!r}. "
            "Retrain, or restore the feature settings the model was trained with."
        )
    shas = shas if shas is not None else [sha1_file(Path(p)) for p in paths]
    X, valid = get_features(cfg, shas, paths, backbone_id=bundle.backbone_id, return_mask=True)
    probs = np.full((len(paths), len(bundle.classes)), np.nan, dtype=np.float64)
    if valid.any():
        if X.shape[1] != bundle.feature_dim:
            raise ValueError(
                f"Feature dim {X.shape[1]} != model dim {bundle.feature_dim}. "
                "Use the backbone/feature settings the model was trained with."
            )
        probs[valid] = bundle.estimator.predict_proba(X[valid])
    return probs, valid
