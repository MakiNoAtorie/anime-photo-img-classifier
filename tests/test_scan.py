"""Manifest building, dedupe and scan report tests (M1 acceptance)."""

from __future__ import annotations

from pathlib import Path

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
from pvd.config import Config
from pvd.scan import scan


def _save(pattern: str, path: Path, seed: int = 0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if pattern == "photo":
        photo_image(seed=seed).save(path)
    else:
        drawing_image(seed=seed).save(path)
    return path


def _row(frame, path: str):
    return frame[frame["path"].str.endswith(path)].iloc[0]


def test_scan_fills_metadata_and_tags(tmp_project: Config) -> None:
    _save("photo", tmp_project.seed_dir / "photo" / "a.png")
    _save("drawing", tmp_project.seed_dir / "drawing" / "b.png")
    _save("photo", tmp_project.pool_dir / "c.jpg", seed=5)
    _save("drawing", tmp_project.test_dir / "drawing" / "realistic" / "d.png", seed=7)

    frame = scan(tmp_project)
    assert set(frame["status"]) == {"ok"}

    photo_row = _row(frame, "seed/photo/a.png")
    assert photo_row["split"] == "seed"
    assert photo_row["folder_label"] == "photo"
    assert photo_row["format"] == "PNG"
    assert photo_row["width"] == 64 and photo_row["height"] == 64
    assert not bool(photo_row["has_alpha"])

    test_row = _row(frame, "test/drawing/realistic/d.png")
    assert test_row["split"] == "test"
    assert test_row["folder_label"] == "drawing"
    assert test_row["tag"] == "realistic"
    assert test_row["source_dir"] == "realistic"


def test_broken_files_recorded_but_do_not_abort(tmp_project: Config) -> None:
    _save("photo", tmp_project.pool_dir / "good.png")
    zero_byte(tmp_project.pool_dir / "zero.jpg")
    truncated_jpeg(tmp_project.pool_dir / "trunc.jpg")

    frame = scan(tmp_project)
    assert _row(frame, "zero.jpg")["status"] == "broken"
    assert _row(frame, "trunc.jpg")["status"] == "broken"
    assert _row(frame, "good.png")["status"] == "ok"


def test_duplicate_prefers_test_over_seed(tmp_project: Config) -> None:
    src = _save("photo", tmp_project.seed_dir / "photo" / "dup.png")
    (tmp_project.test_dir / "photo").mkdir(parents=True, exist_ok=True)
    (tmp_project.test_dir / "photo" / "dup.png").write_bytes(src.read_bytes())

    frame = scan(tmp_project)
    test_row = _row(frame, "test/photo/dup.png")
    seed_row = _row(frame, "seed/photo/dup.png")
    assert test_row["status"] == "ok"
    assert seed_row["status"] == "duplicate"
    assert seed_row["duplicate_of"] == test_row["sha1"]


def test_duplicate_within_pool_marked(tmp_project: Config) -> None:
    src = _save("photo", tmp_project.pool_dir / "one.png")
    (tmp_project.pool_dir / "two.png").write_bytes(src.read_bytes())
    frame = scan(tmp_project)
    statuses = sorted(frame["status"])
    assert statuses == ["duplicate", "ok"]


def test_label_conflict_excludes_both(tmp_project: Config) -> None:
    src = _save("photo", tmp_project.seed_dir / "photo" / "x.png")
    (tmp_project.seed_dir / "drawing").mkdir(parents=True, exist_ok=True)
    (tmp_project.seed_dir / "drawing" / "x.png").write_bytes(src.read_bytes())
    frame = scan(tmp_project)
    assert set(frame["status"]) == {"label_conflict"}


def test_bad_label_directory_flagged(tmp_project: Config) -> None:
    _save("photo", tmp_project.seed_dir / "cats" / "x.png")
    frame = scan(tmp_project)
    assert _row(frame, "seed/cats/x.png")["status"] == "bad_label"


def test_scan_report_written(tmp_project: Config) -> None:
    from pvd.scan import run_scan

    _save("photo", tmp_project.seed_dir / "photo" / "a.png")
    _save("drawing", tmp_project.seed_dir / "drawing" / "b.png")
    report = run_scan(tmp_project)
    assert report.exists()
    text = report.read_text(encoding="utf-8")
    assert "File-format share" in text
    assert tmp_project.manifest_path.exists()


def test_scan_handles_all_edge_case_formats(tmp_project: Config) -> None:
    pool = tmp_project.pool_dir
    rgba_png(pool / "rgba.png")
    palette_transparency_png(pool / "palette.png")
    cmyk_jpeg(pool / "cmyk.jpg")
    gray16_png(pool / "gray16.png")
    animated_gif(pool / "anim.gif")
    exif_rotated_jpeg(pool / "rot.jpg")
    truncated_jpeg(pool / "trunc.jpg")
    zero_byte(pool / "zero.jpg")

    frame = scan(tmp_project)
    by_file = {Path(p).name: row for p, row in zip(frame["path"], frame.to_dict("records"))}
    for name in ("rgba.png", "palette.png", "cmyk.jpg", "gray16.png", "anim.gif", "rot.jpg"):
        assert by_file[name]["status"] == "ok", name
    for name in ("trunc.jpg", "zero.jpg"):
        assert by_file[name]["status"] == "broken", name

