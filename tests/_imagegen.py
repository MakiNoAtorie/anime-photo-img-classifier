"""Synthetic images for hermetic tests (no network, no real photos).

Two easy families keep the ``fake`` backbone separable:
- "drawings": flat pastel fills with hard black outlines,
- "photos": smooth random noise textures.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def drawing_image(size: tuple[int, int] = (64, 64), seed: int = 0) -> Image.Image:
    """Line art on white: a fixed composition with black outlines.

    Kept low-variance so the fake backbone separates the two families
    confidently; a per-seed pixel keeps the bytes unique.
    """
    width, height = size
    array = np.full((height, width, 3), 255, dtype=np.uint8)
    image = Image.fromarray(array)
    draw = ImageDraw.Draw(image)
    jitter = int(seed) % 4
    draw.rectangle([6 + jitter, 6, width - 20, height - 20], outline=(0, 0, 0), width=3)
    draw.ellipse([18, 18, width - 8 - jitter, height - 8], outline=(0, 0, 0), width=3)
    draw.line([0, height - 1, width - 1, height - 1], fill=(0, 0, 0), width=2)
    # Make the bytes unique per seed without changing the appearance much.
    array = np.asarray(image).copy()
    array[0, int(seed) % width] = (int(seed) * 17 + 3) % 256
    return Image.fromarray(array)


def photo_image(size: tuple[int, int] = (64, 64), seed: int = 0) -> Image.Image:
    """A smooth structured gradient with mild sensor-noise.

    Low variance across seeds (same gradient) plus small noise so the same
    class stays tight in embedding space while bytes remain distinct.
    """
    width, height = size
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width]
    base = 128 + 60 * np.sin(xx / 9.0) + 40 * np.cos(yy / 8.0)
    base = base[..., None] + rng.normal(0, 6, (height, width, 3))
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))


def write(path: Path, image: Image.Image, **kwargs) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, **kwargs)
    return path


# -- edge-case writers ------------------------------------------------------ #
def rgba_png(path: Path, color=(255, 0, 0, 0)) -> Path:
    return write(path, Image.new("RGBA", (32, 32), color))


def palette_transparency_png(path: Path) -> Path:
    image = Image.new("P", (32, 32))
    image.putpalette([v for i in range(256) for v in (i, i, i)])
    image.info["transparency"] = 0
    return write(path, image, format="PNG")


def cmyk_jpeg(path: Path) -> Path:
    return write(path, Image.new("CMYK", (32, 32), (10, 20, 30, 0)), format="JPEG")


def gray16_png(path: Path) -> Path:
    rng = np.random.default_rng(1)
    array = (rng.random((32, 32)) * 65535).astype(np.uint16)
    return write(path, Image.fromarray(array), format="PNG")


def truncated_jpeg(path: Path, source: Image.Image | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    source = source or photo_image()
    source.save(path, format="JPEG")
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2])
    return path


def zero_byte(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def animated_gif(path: Path) -> Path:
    frames = [drawing_image(seed=i) for i in range(3)]
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=100, loop=0)
    return path


def exif_rotated_jpeg(path: Path, orientation: int = 6) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = photo_image(size=(40, 20))
    exif = Image.Exif()
    exif[0x0112] = orientation
    image.save(path, format="JPEG", exif=exif)
    return path
