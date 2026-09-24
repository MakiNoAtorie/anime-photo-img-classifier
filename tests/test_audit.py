"""Audit checks: label errors, near-duplicate leakage, shortcuts (M6 acceptance)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from _imagegen import drawing_image, photo_image
from pvd.audit import (
    label_error_suspects,
    near_duplicate_pairs,
    run_audit,
    shortcut_analysis,
)
from pvd.config import Config
from pvd.imageio import sha1_file
from pvd.scan import load_manifest, run_scan


def _fake(cfg: Config) -> None:
    cfg.backbone.id = "fake"
    cfg.num_workers = 0


def test_label_error_suspects_finds_flipped_labels(tmp_project: Config) -> None:
    _fake(tmp_project)
    flipped: set[str] = set()
    for split in ("photo", "drawing"):
        (tmp_project.seed_dir / split).mkdir(parents=True, exist_ok=True)
    # Correct examples.
    for i in range(15):
        photo_image(seed=i).save(tmp_project.seed_dir / "photo" / f"p{i}.png")
        drawing_image(seed=i).save(tmp_project.seed_dir / "drawing" / f"d{i}.png")
    # Deliberately mislabeled examples (drawings in photo, photos in drawing).
    for j in range(3):
        path = tmp_project.seed_dir / "photo" / f"wrong_d{j}.png"
        drawing_image(seed=900 + j).save(path)
        flipped.add(sha1_file(path))
        path = tmp_project.seed_dir / "drawing" / f"wrong_p{j}.png"
        photo_image(seed=900 + j).save(path)
        flipped.add(sha1_file(path))

    run_scan(tmp_project)
    suspects = label_error_suspects(tmp_project)
    top_n = {s["sha1"] for s in suspects[: 2 * len(flipped)]}
    assert len(top_n & flipped) >= len(flipped) / 2


def test_shortcut_probe_flags_format(tmp_project: Config) -> None:
    _fake(tmp_project)
    (tmp_project.seed_dir / "photo").mkdir(parents=True, exist_ok=True)
    (tmp_project.seed_dir / "drawing").mkdir(parents=True, exist_ok=True)
    for i in range(10):
        photo_image(seed=i).save(tmp_project.seed_dir / "photo" / f"p{i}.jpg")
        drawing_image(seed=i).save(tmp_project.seed_dir / "drawing" / f"d{i}.png")
    run_scan(tmp_project)
    shortcuts = shortcut_analysis(tmp_project)
    assert shortcuts["format"]["warn"] is True
    assert shortcuts["format"]["class_from_attribute_accuracy"] >= 0.95


def test_near_duplicate_detection(tmp_project: Config) -> None:
    _fake(tmp_project)
    (tmp_project.seed_dir / "photo").mkdir(parents=True, exist_ok=True)
    (tmp_project.test_dir / "photo").mkdir(parents=True, exist_ok=True)
    base = photo_image(seed=1)
    base.save(tmp_project.seed_dir / "photo" / "orig.png")
    array = np.asarray(base).copy()
    array[0, 0] = (array[0, 0].astype(int) + 1) % 256
    Image.fromarray(array).save(tmp_project.test_dir / "photo" / "near.png")

    run_scan(tmp_project)
    pairs = near_duplicate_pairs(tmp_project)
    assert pairs
    assert pairs[0]["similarity"] > 0.97


def test_run_audit_writes_report(tmp_project: Config) -> None:
    _fake(tmp_project)
    (tmp_project.seed_dir / "photo").mkdir(parents=True, exist_ok=True)
    (tmp_project.seed_dir / "drawing").mkdir(parents=True, exist_ok=True)
    for i in range(8):
        photo_image(seed=i).save(tmp_project.seed_dir / "photo" / f"p{i}.jpg")
        drawing_image(seed=i).save(tmp_project.seed_dir / "drawing" / f"d{i}.png")
    run_scan(tmp_project)
    report = run_audit(tmp_project)
    assert report.exists()
    text = report.read_text(encoding="utf-8")
    assert "Label-error suspects" in text
    assert "Shortcut analysis" in text


def test_review_audit_batch(tmp_project: Config) -> None:
    from pvd.review_app import ReviewSession

    _fake(tmp_project)
    (tmp_project.seed_dir / "photo").mkdir(parents=True, exist_ok=True)
    (tmp_project.seed_dir / "drawing").mkdir(parents=True, exist_ok=True)
    for i in range(15):
        photo_image(seed=i).save(tmp_project.seed_dir / "photo" / f"p{i}.png")
        drawing_image(seed=i).save(tmp_project.seed_dir / "drawing" / f"d{i}.png")
    for j in range(3):
        drawing_image(seed=900 + j).save(tmp_project.seed_dir / "photo" / f"wrong_d{j}.png")
        photo_image(seed=900 + j).save(tmp_project.seed_dir / "drawing" / f"wrong_p{j}.png")
    run_scan(tmp_project)

    session = ReviewSession(tmp_project)
    batch = session.load_audit_batch()
    assert batch
    assert all(item.sampled_as == "audit" for item in batch)
