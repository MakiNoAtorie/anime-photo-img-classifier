"""Predict/sort command tests (M7 acceptance)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from _imagegen import drawing_image, photo_image, zero_byte
from pvd.config import Config
from pvd.predict import run_predict
from pvd.predict import unique_destination
from pvd.probe import run_train
from pvd.scan import run_scan


def _trained_project(cfg: Config) -> None:
    cfg.backbone.id = "fake"
    cfg.num_workers = 0
    for cls, make in (("photo", photo_image), ("drawing", drawing_image)):
        (cfg.seed_dir / cls).mkdir(parents=True, exist_ok=True)
        (cfg.test_dir / cls).mkdir(parents=True, exist_ok=True)
        for i in range(8):
            make(seed=i).save(cfg.seed_dir / cls / f"{cls}{i}.png")
        for i in range(3):
            make(seed=100 + i).save(cfg.test_dir / cls / f"{cls}{i}.png")
    run_scan(cfg)
    run_train(cfg)


def _make_src(tmp_project: Config, n: int = 4) -> Path:
    src = tmp_project.project_root / "inbox"
    src.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        photo_image(seed=300 + i).save(src / f"photo_{i}.png")
        drawing_image(seed=300 + i).save(src / f"drawing_{i}.png")
    zero_byte(src / "broken.jpg")
    return src


def test_unique_destination_avoids_overwrite(tmp_path: Path) -> None:
    target = tmp_path / "a.png"
    target.write_bytes(b"x")
    other = unique_destination(target, "abcdef1234")
    assert other.name == "a_abcdef12.png"
    assert other.exists() is False


def test_predict_copies_into_class_folders(tmp_project: Config) -> None:
    _trained_project(tmp_project)
    src = _make_src(tmp_project)
    dst = tmp_project.project_root / "out"
    csv_path = run_predict(tmp_project, src, dst, mode="copy")

    frame = pd.read_csv(csv_path)
    assert len(frame) == 9  # 8 images + 1 broken
    assert (frame["pred"] == "broken").sum() == 1
    for bucket in ("photo", "drawing", "unsure", "broken"):
        assert (dst / bucket).exists()
    assert (dst / "broken" / "broken.jpg").exists()


def test_predict_never_overwrites(tmp_project: Config) -> None:
    _trained_project(tmp_project)
    src = _make_src(tmp_project)
    dst = tmp_project.project_root / "out"
    run_predict(tmp_project, src, dst, mode="copy")
    first = {p: p.read_bytes() for p in dst.rglob("*.png")}
    run_predict(tmp_project, src, dst, mode="copy")
    # The second run adds new files next to the first run's, never over them.
    assert len(list(dst.rglob("*.png"))) == 2 * len(first)
    assert all(path.read_bytes() == data for path, data in first.items())


def test_predict_move_requires_yes(tmp_project: Config) -> None:
    _trained_project(tmp_project)
    src = _make_src(tmp_project)
    with pytest.raises(ValueError, match="--yes"):
        run_predict(tmp_project, src, tmp_project.project_root / "out", mode="move")

    # With --yes it succeeds and removes the source files.
    run_predict(tmp_project, src, tmp_project.project_root / "out2", mode="move", yes=True)
    assert not list(src.glob("*.png"))


def test_predict_csv_only(tmp_project: Config) -> None:
    _trained_project(tmp_project)
    src = _make_src(tmp_project)
    dst = tmp_project.project_root / "csvout"
    csv_path = run_predict(tmp_project, src, dst, mode="csv_only")
    assert csv_path.exists()
    assert not (dst / "photo").exists()
    frame = pd.read_csv(csv_path)
    assert len(frame) == 9
