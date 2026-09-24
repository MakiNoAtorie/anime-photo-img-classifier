"""Probe training, bundle save/load validation and run_train (M4 acceptance)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from _imagegen import drawing_image, photo_image
from pvd.config import Config
from pvd.probe import (
    ProbeBundle,
    load,
    load_latest,
    quick_retrain,
    run_train,
    save,
    train_probe,
)
from pvd.scan import run_scan


def _separable(n: int = 20, dim: int = 8) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, (n, dim)), rng.normal(3, 1, (n, dim))]).astype(np.float32)
    y = np.array([0] * n + [1] * n)
    return X, y


def test_train_probe_grid_search() -> None:
    X, y = _separable()
    cfg = Config()
    bundle = train_probe(X, y, cfg, classes=["photo", "drawing"], backbone_id="fake")
    assert isinstance(bundle, ProbeBundle)
    assert bundle.best_C in cfg.probe.C_grid
    assert bundle.cv_balanced_accuracy > 0.9
    assert bundle.feature_dim == 8
    assert bundle.n_train == len(y)


def test_save_load_validation(tmp_path: Path) -> None:
    X, y = _separable()
    bundle = train_probe(X, y, Config(), classes=["photo", "drawing"], backbone_id="fake")
    path = save(bundle, tmp_path / "probe.joblib")

    loaded = load(path, expect_backbone_id="fake", expect_dim=8)
    assert loaded.classes == ["photo", "drawing"]

    with pytest.raises(ValueError, match="backbone"):
        load(path, expect_backbone_id="clip_b32")
    with pytest.raises(ValueError, match="expects 8 features"):
        load(path, expect_dim=99)


def test_quick_retrain_matches_C() -> None:
    X, y = _separable()
    estimator = quick_retrain(X, y, C=10.0, config=Config())
    assert estimator.predict(X).tolist() == y.tolist()


def _synthetic_project(cfg: Config, n_seed: int = 8, n_test: int = 4) -> None:
    cfg.backbone.id = "fake"
    cfg.num_workers = 0
    for cls, make in (("photo", photo_image), ("drawing", drawing_image)):
        (cfg.seed_dir / cls).mkdir(parents=True, exist_ok=True)
        (cfg.test_dir / cls).mkdir(parents=True, exist_ok=True)
        for i in range(n_seed):
            make(seed=i).save(cfg.seed_dir / cls / f"{cls}{i}.png")
        for i in range(n_test):
            make(seed=100 + i).save(cfg.test_dir / cls / f"{cls}{i}.png")


def test_run_train_end_to_end(tmp_project: Config) -> None:
    _synthetic_project(tmp_project)
    run_scan(tmp_project)
    path = run_train(tmp_project)
    assert path.exists()

    bundle = load(path, expect_backbone_id="fake")
    assert bundle.label_snapshot_sha
    assert bundle.test_metrics is not None
    assert bundle.test_metrics["balanced_accuracy"] > 0.9
    assert load_latest(tmp_project) == path


def test_run_train_requires_enough_labels(tmp_project: Config) -> None:
    tmp_project.backbone.id = "fake"
    (tmp_project.seed_dir / "photo").mkdir(parents=True, exist_ok=True)
    (tmp_project.seed_dir / "drawing").mkdir(parents=True, exist_ok=True)
    photo_image().save(tmp_project.seed_dir / "photo" / "p.png")
    drawing_image().save(tmp_project.seed_dir / "drawing" / "d.png")
    run_scan(tmp_project)
    with pytest.raises(ValueError, match="Not enough labeled examples"):
        run_train(tmp_project)


def test_run_eval_writes_report(tmp_project: Config) -> None:
    from pvd.evaluate import run_eval

    _synthetic_project(tmp_project)
    run_scan(tmp_project)
    run_train(tmp_project)
    report = run_eval(tmp_project)
    assert report.exists()
    assert "Evaluation report" in report.read_text(encoding="utf-8")
