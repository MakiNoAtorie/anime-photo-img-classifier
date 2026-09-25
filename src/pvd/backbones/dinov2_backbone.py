"""DINOv2 backbone via ``timm`` (self-supervised; no zero-shot) (G23).

DINOv2 is often better than CLIP at low-level texture, which matters for telling
a hyperrealistic drawing from a photo (G24). It has no text encoder, so zero-shot
is unavailable and the tool falls back to "seed with labeled examples".
"""

from __future__ import annotations

import torch

from .base import l2_normalize


class DINOv2Backbone:
    """Self-supervised DINOv2 via timm: strong on texture, no text encoder."""
    is_text_aligned = False

    def __init__(self, backbone_id: str, model_name: str, device: str) -> None:
        import timm

        self.id = backbone_id
        self.model_name = model_name
        self._device = device
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.model.eval().to(device)

        data_config = timm.data.resolve_model_data_config(self.model)
        # A torchvision Compose: picklable, so DataLoader workers can use it (G9).
        self.preprocess = timm.data.create_transform(**data_config, is_training=False)
        self.image_size = int(data_config["input_size"][-1])
        self.dim = int(self.model.num_features)

    def encode(self, batch: torch.Tensor) -> torch.Tensor:
        """Embed a preprocessed batch; returns L2-normalized float32 ``(B, dim)``."""
        batch = batch.to(self._device)
        with torch.inference_mode():
            features = self.model(batch)
        return l2_normalize(features.float())

    def encode_text(self, prompts: list[str]) -> torch.Tensor | None:
        """DINOv2 has no text encoder, so zero-shot is unavailable (returns None)."""
        return None

    def meta(self) -> dict:
        """Serializable identity stored in the embedding cache (model, size, dim)."""
        return {
            "backbone": self.id,
            "library": "timm",
            "model_name": self.model_name,
            "image_size": self.image_size,
            "dim": self.dim,
        }
