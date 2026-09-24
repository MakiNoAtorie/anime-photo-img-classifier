"""Zero-shot prompt ensembles and specialist baseline (M3)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from _imagegen import drawing_image, photo_image
from pvd.zeroshot import class_text_centroids, softmax, zeroshot_probs


class StubTextBackbone:
    """Deterministic text-aligned backbone for unit tests."""

    id = "stub"
    dim = 4
    image_size = 16
    is_text_aligned = True
    logit_scale = 100.0

    def encode_text(self, prompts: list[str]) -> torch.Tensor:
        rows = []
        for prompt in prompts:
            if "photo" in prompt or "photograph" in prompt:
                rows.append([1.0, 0.0, 0.0, 0.0])
            else:
                rows.append([0.0, 1.0, 0.0, 0.0])
        return torch.tensor(rows, dtype=torch.float32)


class NoTextBackbone:
    id = "fake"
    is_text_aligned = False

    def encode_text(self, prompts):
        return None


def test_softmax_rows_sum_to_one() -> None:
    out = softmax(np.array([[1.0, 2.0], [0.0, 0.0]]))
    assert np.allclose(out.sum(axis=1), 1.0)


def test_class_centroids_are_normalized() -> None:
    backbone = StubTextBackbone()
    centroids = class_text_centroids(
        backbone, {"photo": ["a photograph", "a photo"], "drawing": ["a drawing"]}, ["photo", "drawing"]
    )
    assert centroids.shape == (2, 4)
    assert np.allclose(np.linalg.norm(centroids, axis=1), 1.0)


def test_zeroshot_probs_argmax() -> None:
    backbone = StubTextBackbone()
    features = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    probs = zeroshot_probs(
        backbone, features, {"photo": ["a photo"], "drawing": ["a drawing"]}, ["photo", "drawing"]
    )
    assert probs.shape == (2, 2)
    assert probs.argmax(axis=1).tolist() == [0, 1]


def test_centroids_raise_for_missing_prompts() -> None:
    with pytest.raises(ValueError, match="No zero-shot prompts"):
        class_text_centroids(StubTextBackbone(), {"photo": ["a photo"]}, ["photo", "drawing"])


def test_centroids_raise_for_textless_backbone() -> None:
    with pytest.raises(ValueError, match="not text-aligned"):
        class_text_centroids(NoTextBackbone(), {"photo": ["x"], "drawing": ["y"]}, ["photo", "drawing"])


def test_run_zeroshot_rejects_textless_backbone(tmp_project) -> None:
    from pvd.scan import run_scan
    from pvd.zeroshot import run_zeroshot

    tmp_project.backbone.id = "fake"
    (tmp_project.test_dir / "photo").mkdir(parents=True, exist_ok=True)
    (tmp_project.test_dir / "drawing").mkdir(parents=True, exist_ok=True)
    photo_image(seed=1).save(tmp_project.test_dir / "photo" / "p.png")
    drawing_image(seed=1).save(tmp_project.test_dir / "drawing" / "d.png")
    run_scan(tmp_project)
    with pytest.raises(ValueError, match="no text encoder"):
        run_zeroshot(tmp_project)


def test_specialist_baseline_report(tmp_project) -> None:
    from pvd.evaluate import run_specialist_baseline
    from pvd.scan import run_scan

    (tmp_project.test_dir / "photo").mkdir(parents=True, exist_ok=True)
    (tmp_project.test_dir / "drawing").mkdir(parents=True, exist_ok=True)
    for i in range(3):
        photo_image(seed=i).save(tmp_project.test_dir / "photo" / f"p{i}.png")
        drawing_image(seed=i).save(tmp_project.test_dir / "drawing" / f"d{i}.png")
    run_scan(tmp_project)

    def fake_score(path: str) -> float:
        return 0.9 if "photo" in path else 0.1

    report = run_specialist_baseline(tmp_project, "fake_specialist", fake_score)
    assert report.exists()
    text = report.read_text(encoding="utf-8")
    assert "fake_specialist" in text
    assert "100.00%" in text
