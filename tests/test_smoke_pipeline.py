"""End-to-end smoke test: scan -> embed(fake) -> train -> predict (section 13)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from _imagegen import drawing_image, photo_image
from pvd.audit import run_audit
from pvd.config import Config
from pvd.evaluate import run_eval
from pvd.features import run_embed
from pvd.predict import run_predict
from pvd.models import load_any as load
from pvd.probe import run_train
from pvd.scan import load_manifest, run_scan


def _seed(root: Path, folder: str, make, n: int, prefix: str, start: int) -> None:
    """Distinct seeds per split: identical bytes would be deduped (or leak)."""
    (root / folder).mkdir(parents=True, exist_ok=True)
    for i in range(n):
        make(seed=start + i).save(root / folder / f"{prefix}_{folder}_{i}.png")


def test_full_pipeline(tmp_project: Config) -> None:
    cfg = tmp_project
    cfg.backbone.id = "fake"
    cfg.num_workers = 0

    _seed(cfg.seed_dir, "photo", photo_image, 20, "s", start=1000)
    _seed(cfg.seed_dir, "drawing", drawing_image, 20, "s", start=1000)
    _seed(cfg.test_dir, "photo", photo_image, 5, "t", start=2000)
    _seed(cfg.test_dir, "drawing", drawing_image, 5, "t", start=2000)

    run_scan(cfg)
    manifest = load_manifest(cfg)
    assert (manifest["status"] == "ok").all()  # no accidental duplicates
    run_embed(cfg)
    bundle_path = run_train(cfg)
    bundle = load(bundle_path)
    assert bundle.test_metrics is not None
    assert bundle.test_metrics["balanced_accuracy"] > 0.9

    assert run_eval(cfg).exists()

    src = cfg.project_root / "inbox"
    _seed(src, "photo", photo_image, 5, "new", start=3000)
    _seed(src, "drawing", drawing_image, 5, "new", start=3000)
    csv_path = run_predict(cfg, src, cfg.project_root / "sorted", mode="copy")
    frame = pd.read_csv(csv_path)
    assert len(frame) == 10
    expected = np.where(frame["path"].str.contains("drawing"), "drawing", "photo")
    assert (frame["pred"].values == expected).mean() > 0.9

    assert run_audit(cfg).exists()
