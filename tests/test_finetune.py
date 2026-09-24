"""Fine-tuning stage tests (M9 acceptance). Uses a tiny random-init timm model."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from _imagegen import drawing_image, photo_image
from pvd.config import Config
from pvd.finetune import finetune_predict_proba, run_finetune
from pvd.models import load_any
from pvd.probe import run_train
from pvd.scan import run_scan


def _project(cfg: Config) -> None:
    cfg.backbone.id = "fake"
    cfg.num_workers = 0
    cfg.finetune.model_name = "resnet18"
    cfg.finetune.pretrained = False
    cfg.finetune.image_size = 32
    cfg.finetune.epochs = 1
    cfg.finetune.batch_size = 4
    for cls, make in (("photo", photo_image), ("drawing", drawing_image)):
        (cfg.seed_dir / cls).mkdir(parents=True, exist_ok=True)
        (cfg.test_dir / cls).mkdir(parents=True, exist_ok=True)
        for i in range(8):
            make(seed=i).save(cfg.seed_dir / cls / f"{cls}{i}.png")
        for i in range(4):
            make(seed=100 + i).save(cfg.test_dir / cls / f"{cls}{i}.png")


def test_finetune_trains_saves_and_predicts(tmp_project: Config) -> None:
    _project(tmp_project)
    run_scan(tmp_project)
    run_train(tmp_project)  # probe for the comparison section

    path = run_finetune(tmp_project)
    assert path.exists()
    bundle = load_any(path)
    assert bundle.kind == "finetune"
    assert bundle.classes == ["photo", "drawing"]
    assert tmp_project.resolve(bundle.checkpoint).exists()

    paths = [str(p) for p in tmp_project.test_dir.rglob("*.png")]
    probs, valid = finetune_predict_proba(bundle, paths, tmp_project)
    assert valid.all()
    assert probs.shape == (len(paths), 2)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_predict_and_eval_accept_finetune_bundle(tmp_project: Config) -> None:
    from pvd.evaluate import run_eval
    from pvd.predict import run_predict

    _project(tmp_project)
    run_scan(tmp_project)
    run_train(tmp_project)
    run_finetune(tmp_project)

    # `run_eval`/`run_predict` default to the newest bundle (the fine-tune).
    report = run_eval(tmp_project)
    assert report.exists()
    assert "Kind: `finetune`" in report.read_text(encoding="utf-8")

    src = tmp_project.project_root / "inbox"
    src.mkdir(parents=True, exist_ok=True)
    for i in range(3):
        photo_image(seed=500 + i).save(src / f"p{i}.png")
        drawing_image(seed=500 + i).save(src / f"d{i}.png")
    csv_path = run_predict(tmp_project, src, tmp_project.project_root / "out", mode="copy")
    assert csv_path.exists()
    assert len(csv_path.read_text(encoding="utf-8").strip().splitlines()) == 7  # header + 6
