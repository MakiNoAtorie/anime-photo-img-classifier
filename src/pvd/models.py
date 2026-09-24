"""Unified model loading and prediction across probe and fine-tuned bundles.

The probe predicts from cached embeddings; the fine-tuned model predicts from
raw images. Callers (``predict``, ``eval``) should not care which they hold, so
this module hides the difference behind :func:`predict_proba_paths`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np

from .config import Config
from .features import get_features
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


def load_latest_any(config: Config) -> Path:
    """Newest bundle of any supported kind, or a clear error."""
    candidates = sorted(config.models_dir.glob("*.joblib"), key=lambda p: p.stat().st_mtime)
    for path in reversed(candidates):
        try:
            bundle = joblib.load(path)
        except Exception:  # noqa: BLE001
            continue
        if getattr(bundle, "kind", "probe") in KNOWN_KINDS:
            return path
    raise FileNotFoundError(f"No model bundle found in {config.models_dir}. Run `pvd train` first.")


def predict_proba_paths(
    bundle,
    paths: list[str],
    config: Config,
    shas: list[str] | None = None,
) -> np.ndarray:
    """Return ``(N, C)`` class probabilities for image paths."""
    kind = getattr(bundle, "kind", "probe")
    if kind == "finetune":
        from .finetune import finetune_predict_proba

        return finetune_predict_proba(bundle, paths, config)

    shas = shas if shas is not None else [sha1_file(Path(p)) for p in paths]
    X = get_features(config, shas, paths, backbone_id=bundle.backbone_id)
    if X.shape[1] != bundle.feature_dim:
        raise ValueError(
            f"Feature dim {X.shape[1]} != model dim {bundle.feature_dim}. "
            "Use the backbone/feature settings the model was trained with."
        )
    return bundle.estimator.predict_proba(X)
