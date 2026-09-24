"""Zero-shot baseline using prompt ensembles (G25).

Until enough labels exist, this is the tool's only classifier. Several prompts
per class are embedded and averaged into a text *centroid*; the image embedding
is compared against each centroid. Only the global feature block is used (tiles
and extra features are not text-aligned).

Note: softmax over CLIP/SigLIP similarities is **not calibrated**. Treat the
probabilities as rankings, and pick any decision threshold from the test-set
coverage curve, not from intuition.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .config import Config
from .device import pick_device
from .evaluate import evaluate_arrays, render_report, resolve_test_set, write_report
from .features import get_features

log = logging.getLogger(__name__)


def class_text_centroids(backbone, prompts_by_class: dict[str, list[str]], classes: list[str]) -> np.ndarray:
    """Average each class's prompt embeddings into an L2-normalized centroid."""
    rows = []
    for name in classes:
        prompts = prompts_by_class.get(name)
        if not prompts:
            raise ValueError(f"No zero-shot prompts configured for class {name!r}")
        embeddings = backbone.encode_text(list(prompts))
        if embeddings is None:
            raise ValueError(
                f"Backbone {backbone.id!r} is not text-aligned; zero-shot is unavailable. "
                "Seed the classifier with labeled examples instead."
            )
        centroid = embeddings.mean(dim=0)
        rows.append(centroid.detach().cpu().numpy())
    centroids = np.stack(rows).astype(np.float32)
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True).clip(min=1e-12)
    return centroids


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def zeroshot_probs(
    backbone,
    features: np.ndarray,
    prompts_by_class: dict[str, list[str]],
    classes: list[str],
) -> np.ndarray:
    """Return ``(N, C)`` probabilities from prompt-centroid similarity."""
    centroids = class_text_centroids(backbone, prompts_by_class, classes)
    scale = getattr(backbone, "logit_scale", None) or 100.0
    logits = (features @ centroids.T) * scale
    return softmax(logits)


def run_zeroshot(config: Config) -> Path:
    """Predict the test set zero-shot and write a report."""
    from .backbones import build_backbone

    # Resolve the test set first so missing data fails fast, before model loads.
    test = resolve_test_set(config)

    device = pick_device(config.device)
    backbone = build_backbone(config.backbone.id, device)
    if not getattr(backbone, "is_text_aligned", False):
        raise ValueError(
            f"Backbone {config.backbone.id!r} has no text encoder, so `pvd zeroshot` "
            "cannot run. Choose a CLIP/SigLIP backbone or use `pvd train` with seeds."
        )

    features = get_features(config, test.shas, test.paths)
    probs = zeroshot_probs(backbone, features, config.zeroshot.prompts, config.classes)

    result = evaluate_arrays(test.y_true, probs, list(config.classes), test.frame, test.tags)
    extra = [f"Backbone: `{config.backbone.id}` | prompts per class: "
             + ", ".join(f"{c}={len(config.zeroshot.prompts.get(c, []))}" for c in config.classes), ""]
    report = write_report(config, "zeroshot", render_report("zero-shot", result, extra))
    log.info("Zero-shot balanced accuracy: %.3f", result.balanced_accuracy)
    return report
