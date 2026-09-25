"""Robust image loading tests (M1 acceptance)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from _imagegen import (
    animated_gif,
    cmyk_jpeg,
    drawing_image,
    exif_rotated_jpeg,
    gray16_png,
    palette_transparency_png,
    photo_image,
    rgba_png,
    truncated_jpeg,
    zero_byte,
)
from pvd.imageio import ImageLoadError, load_rgb, probe_image, sha1_file


def test_sha1_is_content_addressed(tmp_path: Path) -> None:
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"hello")
    b.write_bytes(b"hello")
    assert sha1_file(a) == sha1_file(b)
    b.write_bytes(b"hello!")
    assert sha1_file(a) != sha1_file(b)


def test_rgba_composited_on_background(tmp_path: Path) -> None:
    # Fully transparent red -> should become the background color.
    path = rgba_png(tmp_path / "t.png", (255, 0, 0, 0))
    image = load_rgb(path, bg=(255, 255, 255))
    assert image.mode == "RGB"
    assert image.getpixel((5, 5)) == (255, 255, 255)

    info = probe_image(path)
    assert info.has_alpha is True
    assert info.mode == "RGBA"


def test_palette_transparency_detected(tmp_path: Path) -> None:
    path = palette_transparency_png(tmp_path / "p.png")
    info = probe_image(path)
    assert info.has_alpha is True
    assert load_rgb(path).mode == "RGB"


def test_cmyk_converted(tmp_path: Path) -> None:
    path = cmyk_jpeg(tmp_path / "c.jpg")
    assert load_rgb(path).mode == "RGB"


def test_16bit_grayscale_scaled(tmp_path: Path) -> None:
    path = gray16_png(tmp_path / "g.png")
    info = probe_image(path)
    assert info.mode in ("I;16", "I;16B", "I;16L")
    image = load_rgb(path)
    assert image.mode == "RGB"
    # Uniform 16-bit noise must map onto the full 8-bit range. Naive clipping
    # (Pillow's plain convert) gives mean ~255 and std ~4.
    array = np.asarray(image)
    assert array.min() <= 5 and array.max() >= 250
    assert 110 < array.mean() < 145


def test_truncated_jpeg_raises_broken(tmp_path: Path) -> None:
    path = truncated_jpeg(tmp_path / "t.jpg")
    with pytest.raises(ImageLoadError) as exc:
        load_rgb(path)
    assert exc.value.reason == "broken"


def test_zero_byte_raises_broken(tmp_path: Path) -> None:
    path = zero_byte(tmp_path / "z.jpg")
    with pytest.raises(ImageLoadError) as exc:
        load_rgb(path)
    assert exc.value.reason == "broken"


def test_animated_gif_uses_first_frame(tmp_path: Path) -> None:
    path = animated_gif(tmp_path / "a.gif")
    info = probe_image(path)
    assert info.n_frames == 3
    image = np.asarray(load_rgb(path))
    # Each synthetic frame marks one pixel in row 0: frame 0 marks x=0 (value 3),
    # frame 1 marks x=1 (value 20). Frame 0 must be the one loaded.
    assert tuple(image[0, 0]) == (3, 3, 3)
    assert tuple(image[0, 1]) == (255, 255, 255)


def test_exif_orientation_swaps_dimensions(tmp_path: Path) -> None:
    path = exif_rotated_jpeg(tmp_path / "r.jpg", orientation=6)
    info = probe_image(path)
    # Stored 40x20, orientation 6 => reported as portrait 20x40.
    assert (info.width, info.height) == (20, 40)
    assert load_rgb(path).size == (20, 40)


def test_too_large_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "big.png"
    drawing_image(size=(64, 64)).save(path)
    with pytest.raises(ImageLoadError) as exc:
        load_rgb(path, max_pixels=100)
    assert exc.value.reason == "too_large"


def test_probe_does_not_require_decode(tmp_path: Path) -> None:
    path = tmp_path / "photo.png"
    photo_image(size=(30, 50)).save(path)
    info = probe_image(path)
    assert (info.width, info.height) == (30, 50)
    assert info.format == "PNG"
