"""Deterministic fake backbone for hermetic tests (section 13).

It downsamples to 16x16, flattens, applies a fixed seeded random projection and
L2-normalizes. It downloads nothing and is deterministic across runs, yet the
linear structure is enough to separate the synthetic "photo" (noise texture) and
"drawing" (flat shapes + outlines) families used in tests.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from PIL import Image

from .base import l2_normalize


class _FakePreprocess:
    """Picklable preprocessing (resize, to float tensor in [0, 1])."""

    def __init__(self, image_size: int) -> None:
        self.image_size = image_size

    def __call__(self, image: Image.Image) -> torch.Tensor:
        resized = image.convert("RGB").resize(
            (self.image_size, self.image_size), Image.BILINEAR
        )
        array = torch.frombuffer(bytearray(resized.tobytes()), dtype=torch.uint8)
        array = array.reshape(self.image_size, self.image_size, 3).permute(2, 0, 1)
        return array.float() / 255.0


class FakeBackbone:
    """Random projection of downsampled pixels; tests only."""

    id = "fake"
    image_size = 16
    is_text_aligned = False

    def __init__(self, dim: int = 128, seed: int = 1234) -> None:
        self.dim = dim
        self.preprocess = _FakePreprocess(self.image_size)
        generator = torch.Generator().manual_seed(seed)
        in_features = 3 * self.image_size * self.image_size
        self._projection = torch.randn(in_features, dim, generator=generator)
        self._bias = torch.randn(dim, generator=generator)

    def to(self, device: str) -> "FakeBackbone":  # pragma: no cover - trivial
        """Move the projection to ``device`` (mirrors the real backbones' API)."""
        self._projection = self._projection.to(device)
        self._bias = self._bias.to(device)
        return self

    def encode(self, batch: torch.Tensor) -> torch.Tensor:
        """Project flattened pixels and L2-normalize; deterministic across runs."""
        flat = batch.flatten(start_dim=1)
        projected = flat @ self._projection.to(flat.device) + self._bias.to(flat.device)
        return l2_normalize(F.relu(projected))

    def encode_text(self, prompts: list[str]) -> torch.Tensor | None:
        """The plain fake backbone is not text-aligned (returns None)."""
        return None

    def meta(self) -> dict:
        """Serializable identity stored in the embedding cache."""
        return {"backbone": self.id, "dim": self.dim}


class FakeTextBackbone(FakeBackbone):
    """``fake`` plus a deterministic text encoder, so zero-shot code paths can
    be tested without downloading CLIP. Prompts map to seeded random unit
    vectors (a hash of the text picks the seed); tests only.
    """

    id = "fake_text"
    is_text_aligned = True
    logit_scale = 100.0

    def encode_text(self, prompts: list[str]) -> torch.Tensor:
        """Deterministic unit vectors per prompt, so zero-shot code runs offline."""
        rows = []
        for prompt in prompts:
            seed = sum(ord(ch) * (i + 1) for i, ch in enumerate(prompt)) % (2**31)
            generator = torch.Generator().manual_seed(seed)
            rows.append(torch.randn(self.dim, generator=generator))
        return l2_normalize(torch.stack(rows))
