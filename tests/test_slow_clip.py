"""Real-model check (downloads ~600 MB of CLIP weights on first run).

Deselected by default; run with ``pytest -m slow``. It proves the real
``clip_b32`` backbone loads, embeds and does prompt-ensemble zero-shot on the
synthetic images well above chance.
"""

from __future__ import annotations

import pytest

from _imagegen import drawing_image, photo_image
from pvd.config import Config
from pvd.scan import run_scan

pytestmark = pytest.mark.slow


def test_clip_b32_zeroshot_end_to_end(tmp_project: Config) -> None:
    pytest.importorskip("open_clip")
    from pvd.evaluate import resolve_test_set
    from pvd.features import get_features, global_block, load_feature_meta
    from pvd.backbones import build_backbone
    from pvd.zeroshot import zeroshot_probs

    cfg = tmp_project
    cfg.backbone.id = "clip_b32"
    cfg.num_workers = 0
    cfg.device = "cpu"
    for cls, make in (("photo", photo_image), ("drawing", drawing_image)):
        (cfg.test_dir / cls).mkdir(parents=True, exist_ok=True)
        for i in range(6):
            make(size=(224, 224), seed=i).save(cfg.test_dir / cls / f"{cls}{i}.png")
    run_scan(cfg)

    test = resolve_test_set(cfg)
    features, valid = get_features(cfg, test.shas, test.paths, return_mask=True)
    assert valid.all()
    assert features.shape == (12, 512)
    backbone = build_backbone("clip_b32", "cpu")
    probs = zeroshot_probs(backbone, global_block(features, load_feature_meta(cfg)), cfg.zeroshot.prompts, cfg.classes)
    accuracy = float((probs.argmax(axis=1) == test.y_true).mean())
    assert accuracy >= 0.75
