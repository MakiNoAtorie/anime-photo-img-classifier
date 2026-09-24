"""Backbone registry: maps config ids to concrete models.

Heavy model construction is cached per (id, device) so it happens once per
process. Import of ``open_clip``/``timm`` happens inside :func:`build_backbone`
so commands that never embed (e.g. ``pvd audit --help``) stay lightweight.
"""

from __future__ import annotations

from functools import lru_cache

from .base import Backbone

# id -> (family, model_name, pretrained_tag)
MODEL_REGISTRY: dict[str, tuple[str, str, str | None]] = {
    "clip_b32": ("clip", "ViT-B-32", "laion2b_s34b_b79k"),
    "clip_l14": ("clip", "ViT-L-14", "laion2b_s32b_b82k"),
    "siglip_so400m": ("clip", "ViT-SO400M-14-SigLIP", "webli"),
    "dinov2_s": ("dinov2", "vit_small_patch14_dinov2", None),
    "dinov2_b": ("dinov2", "vit_base_patch14_dinov2", None),
}

FAKE_ID = "fake"
FAKE_TEXT_ID = "fake_text"  # tests only: fake + deterministic text encoder


@lru_cache(maxsize=None)
def build_backbone(backbone_id: str, device: str = "cpu") -> Backbone:
    """Construct (once) and return the backbone for ``backbone_id``."""
    if backbone_id == FAKE_ID:
        from .fake_backbone import FakeBackbone

        return FakeBackbone().to(device)
    if backbone_id == FAKE_TEXT_ID:
        from .fake_backbone import FakeTextBackbone

        return FakeTextBackbone().to(device)

    if backbone_id not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown backbone {backbone_id!r}. Known: {sorted(MODEL_REGISTRY) + [FAKE_ID, FAKE_TEXT_ID]}"
        )

    family, model_name, pretrained = MODEL_REGISTRY[backbone_id]
    if family == "clip":
        from .clip_backbone import ClipBackbone

        return ClipBackbone(backbone_id, model_name, pretrained or "", device)
    if family == "dinov2":
        from .dinov2_backbone import DINOv2Backbone

        return DINOv2Backbone(backbone_id, model_name, device)
    raise ValueError(f"Unsupported backbone family: {family}")


def backbone_dim(backbone_id: str, device: str = "cpu") -> int:
    """Feature dimension of a backbone (loads the model if necessary)."""
    return build_backbone(backbone_id, device).dim


__all__ = ["Backbone", "MODEL_REGISTRY", "FAKE_ID", "build_backbone", "backbone_dim"]
