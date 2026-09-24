"""Robust image loading and hashing (gaps G3, G4, G5, G7).

Every image entering the pipeline goes through here. The loader is deliberately
defensive: a single corrupt file in a large folder must never abort a run, so it
raises a typed :class:`ImageLoadError` that callers can record in the manifest
and skip.

Decoding rules, in order:

1. Open with Pillow; use frame 0 for multi-frame files.
2. Enforce ``max_pixels`` (decompression-bomb guard).
3. Apply ``ImageOps.exif_transpose`` so rotated phone photos are upright (G4).
4. Composite alpha over a fixed neutral background, recording transparency as a
   separate manifest attribute so it can be audited as a shortcut (G5).
5. Convert every remaining mode (CMYK, 16-bit, palette, float, ...) to RGB.
6. ``load()`` inside the guarded block so truncated files fail here, not later.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

# Optional HEIC support for iPhone photos. Import error is not fatal.
try:  # pragma: no cover - depends on optional package
    import pillow_heif

    pillow_heif.register_heif_opener()
except Exception:  # pragma: no cover
    pass

# Pillow's own decompression-bomb guard stays at its default (it errors above
# 2x MAX_IMAGE_PIXELS). The project's configurable, stricter limit
# (``image.max_pixels``) is checked from the header by ``_enforce_size`` before
# any pixels are decoded, so it never needs to change Pillow's global state.

# Decoders Pillow may use. Pillow picks a decoder from the file *content*, not
# the extension, so without this list a file named `x.jpg` could reach rarely
# used decoders (EPS hands the file to Ghostscript, PSD has had memory bugs).
# JPEG also covers MPO (multi-picture JPEGs from phones); HEIF is added only
# when the optional pillow-heif opener is installed.
_ALLOWED_FORMATS = ["JPEG", "PNG", "WEBP", "BMP", "GIF", "TIFF"]


def allowed_formats() -> list[str]:
    """Pillow format names this project will decode."""
    extra = [name for name in ("HEIF",) if name in Image.OPEN]
    return _ALLOWED_FORMATS + extra


# Modes that carry an alpha channel.
_ALPHA_MODES = {"RGBA", "LA", "PA", "RGBa", "La"}


class ImageLoadError(Exception):
    """Raised when an image cannot be decoded into RGB."""

    def __init__(self, reason: str, message: str = "") -> None:
        self.reason = reason
        self.message = message or reason
        super().__init__(self.message)


@dataclass(frozen=True)
class ImageInfo:
    """Metadata read from an image header (no pixel conversion)."""

    width: int
    height: int
    format: str
    mode: str
    has_alpha: bool
    n_frames: int


def sha1_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the SHA-1 hex digest of a file's bytes.

    Content addressing by bytes (not path) is what lets caches survive renames
    and folder moves (G7).
    """
    digest = hashlib.sha1()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _has_alpha(im: Image.Image) -> bool:
    if im.mode in _ALPHA_MODES:
        return True
    if im.mode == "P" and "transparency" in im.info:
        return True
    return False


def _read_info(im: Image.Image) -> ImageInfo:
    width, height = im.size
    # EXIF orientation 5-8 swap width and height.
    orientation = 1
    try:
        orientation = im.getexif().get(0x0112, 1)
    except Exception:
        orientation = 1
    if orientation in (5, 6, 7, 8):
        width, height = height, width
    return ImageInfo(
        width=width,
        height=height,
        format=im.format or "UNKNOWN",
        mode=im.mode,
        has_alpha=_has_alpha(im),
        n_frames=int(getattr(im, "n_frames", 1)),
    )


def _enforce_size(im: Image.Image, max_pixels: int) -> None:
    if max_pixels and im.size[0] * im.size[1] > max_pixels:
        raise ImageLoadError(
            "too_large",
            f"{im.size[0]}x{im.size[1]} exceeds max_pixels={max_pixels}",
        )


def probe_image(path: Path) -> ImageInfo:
    """Read image metadata without decoding pixels.

    Used by ``scan`` to fill the manifest cheaply. Integrity is checked
    separately by :func:`load_rgb` when a full decode is required.
    """
    try:
        with Image.open(path, formats=allowed_formats()) as im:
            return _read_info(im)
    except Image.DecompressionBombError as exc:
        raise ImageLoadError("too_large", str(exc)) from exc
    except ImageLoadError:
        raise
    except Exception as exc:  # PIL raises many exception types
        raise ImageLoadError("broken", f"{type(exc).__name__}: {exc}") from exc


def _scaled_array_to_rgb(arr: np.ndarray) -> Image.Image:
    """Convert a non-8-bit numeric array to 8-bit RGB without clipping."""
    if arr.dtype == np.uint16:
        # 65535 / 257 == 255 exactly, so this maps the full range to 8-bit.
        scaled = (arr // 257).astype(np.uint8)
    else:
        data = arr.astype(np.float32)
        low, high = float(data.min()), float(data.max())
        if high > low:
            data = (data - low) / (high - low) * 255.0
        else:
            data = np.zeros_like(data)
        scaled = data.astype(np.uint8)
    return Image.fromarray(scaled).convert("RGB")


def _to_rgb(im: Image.Image, bg: tuple[int, int, int]) -> Image.Image:
    """Convert any mode to RGB, compositing alpha over ``bg``."""
    if _has_alpha(im):
        rgba = im.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (*bg, 255))
        return Image.alpha_composite(background, rgba).convert("RGB")
    if im.mode == "RGB":
        return im
    if im.mode in ("L", "1"):
        # Let Pillow dither 1-bit images into grayscale, then promote to RGB.
        return im.convert("RGB")
    if im.mode in ("I;16", "I;16L", "I;16B", "I;16N", "I", "F"):
        return _scaled_array_to_rgb(np.asarray(im))
    # CMYK, P (without transparency), YCbCr, LAB, HSV, ...
    return im.convert("RGB")


def load_rgb(
    path: Path,
    bg: tuple[int, int, int] = (255, 255, 255),
    max_pixels: int = 80_000_000,
) -> Image.Image:
    """Decode ``path`` to an upright RGB image.

    Raises :class:`ImageLoadError` with ``reason`` in
    ``{'broken', 'too_large'}`` on failure. The returned image is fully loaded
    and detached from the file handle.
    """
    try:
        with Image.open(path, formats=allowed_formats()) as im:
            _enforce_size(im, max_pixels)
            try:
                im.seek(0)  # first frame of animated GIFs/WebP
            except (EOFError, AttributeError):
                pass
            im = ImageOps.exif_transpose(im)
            rgb = _to_rgb(im, bg)
            rgb.load()  # force decode so truncated files fail here
            return rgb
    except Image.DecompressionBombError as exc:
        raise ImageLoadError("too_large", str(exc)) from exc
    except ImageLoadError:
        raise
    except Exception as exc:
        raise ImageLoadError("broken", f"{type(exc).__name__}: {exc}") from exc
