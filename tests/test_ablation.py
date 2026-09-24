"""Ablation report generation (M8 acceptance)."""

from __future__ import annotations

from pathlib import Path

from _imagegen import drawing_image, photo_image
from pvd.ablation import AblationSpec, run_ablation
from pvd.config import Config
from pvd.scan import run_scan


def _project(cfg: Config) -> None:
    cfg.backbone.id = "fake"
    cfg.num_workers = 0
    for cls, make in (("photo", photo_image), ("drawing", drawing_image)):
        (cfg.seed_dir / cls).mkdir(parents=True, exist_ok=True)
        (cfg.test_dir / cls).mkdir(parents=True, exist_ok=True)
        for i in range(8):
            make(seed=i).save(cfg.seed_dir / cls / f"{cls}{i}.png")
        for i in range(3):
            make(seed=100 + i).save(cfg.test_dir / cls / f"{cls}{i}.png")


def test_ablation_table_and_recommendation(tmp_project: Config) -> None:
    _project(tmp_project)
    run_scan(tmp_project)
    specs = [
        AblationSpec("fake", "fake", tiles=False),
        AblationSpec("fake+tiles5", "fake", tiles=True),
    ]
    report = run_ablation(tmp_project, specs=specs)
    assert report.exists()
    text = report.read_text(encoding="utf-8")
    assert "Ablation report" in text
    assert "fake" in text
    assert "Recommendation" in text
    # Both rows completed without error.
    assert "| fake |" in text
