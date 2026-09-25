"""CLIP / SigLIP backbones via ``open_clip`` (G23, G25).

The model is created once per process by the registry's cache. Text encoding is
exposed so :mod:`pvd.zeroshot` can build prompt ensembles.
"""

from __future__ import annotations

import torch

from .base import l2_normalize


def _normalize_image_size(size: object) -> int:
    if isinstance(size, (tuple, list)):
        return int(size[0])
    return int(size)


class ClipBackbone:
    """A text-aligned OpenCLIP model (CLIP or SigLIP)."""

    is_text_aligned = True

    def __init__(self, backbone_id: str, model_name: str, pretrained: str, device: str) -> None:
        import open_clip

        self.id = backbone_id
        self.model_name = model_name
        self.pretrained = pretrained
        self._device = device
        self.model, _, transform = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.preprocess = transform  # torchvision Compose: picklable (G9)
        self.image_size = _normalize_image_size(self.model.visual.image_size)
        self.dim = self._infer_dim()
        scale = getattr(self.model, "logit_scale", None)
        self.logit_scale = float(scale.exp().item()) if scale is not None else None

    def _infer_dim(self) -> int:
        output_dim = getattr(self.model.visual, "output_dim", None)
        if isinstance(output_dim, int):
            return output_dim
        projection = getattr(self.model, "text_projection", None)
        if projection is not None and hasattr(projection, "shape"):
            return int(projection.shape[-1])
        raise RuntimeError(f"Could not infer embedding dim for {self.model_name}")

    def encode(self, batch: torch.Tensor) -> torch.Tensor:
        """Embed a preprocessed batch; returns L2-normalized float32 ``(B, dim)``."""
        batch = batch.to(self._device)
        with torch.inference_mode():
            features = self.model.encode_image(batch)
        return l2_normalize(features.float())

    def encode_text(self, prompts: list[str]) -> torch.Tensor | None:
        """Embed prompts into the same space as images (used for zero-shot)."""
        tokens = self.tokenizer(prompts).to(self._device)
        with torch.inference_mode():
            features = self.model.encode_text(tokens)
        return l2_normalize(features.float())

    def meta(self) -> dict:
        """Serializable identity stored in the embedding cache (model, weights, size)."""
        return {
            "backbone": self.id,
            "library": "open_clip",
            "model_name": self.model_name,
            "pretrained": self.pretrained,
            "image_size": self.image_size,
            "dim": self.dim,
        }
