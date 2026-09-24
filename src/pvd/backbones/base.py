"""Backbone interface.

A backbone turns an image into a fixed-length embedding. The protocol deliberately
keeps text encoding optional: CLIP/SigLIP can do zero-shot, DINOv2 cannot (G23).

``preprocess`` is a **picklable callable object**, not a bound method. This is
what lets the embedding ``DataLoader`` use worker processes on Windows/macOS
(spawn start method), where anything passed to a worker must be picklable and a
model cannot be (G9).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
from PIL import Image


@dataclass
class TransformPreprocess:
    """Picklable wrapper around a torchvision/timm transform.

    Keeping this a standalone object (rather than a bound method on a model)
    lets the embedding DataLoader use worker processes under the spawn start
    method (G9).
    """

    transform: object

    def __call__(self, image: Image.Image) -> torch.Tensor:
        return self.transform(image)


def l2_normalize(tensor: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """L2-normalize each row. Used identically at train and predict time (NF4)."""
    return tensor / tensor.norm(dim=-1, keepdim=True).clamp_min(eps)


@runtime_checkable
class Backbone(Protocol):
    """Minimal interface every embedding model must satisfy."""

    id: str
    dim: int
    image_size: int
    is_text_aligned: bool
    #: Picklable callable: PIL.Image -> torch.Tensor of shape (3, H, W).
    preprocess: object

    def encode(self, batch: torch.Tensor) -> torch.Tensor:
        """Encode a preprocessed batch, returning L2-normalized (B, dim)."""

    def encode_text(self, prompts: list[str]) -> torch.Tensor | None:
        """Return L2-normalized text embeddings, or ``None`` if unsupported."""

    def meta(self) -> dict:
        """Serializable identity of the loaded model, stored in the cache."""
