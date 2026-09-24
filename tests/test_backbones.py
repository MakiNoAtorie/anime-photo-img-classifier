"""Backbone registry and fake backbone tests (M2)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pvd.backbones import build_backbone
from pvd.backbones.fake_backbone import FakeBackbone


def test_fake_backbone_shapes_and_normalization() -> None:
    backbone = build_backbone("fake", "cpu")
    assert backbone.dim == 128
    assert backbone.image_size == 16
    assert backbone.is_text_aligned is False

    from _imagegen import drawing_image

    tensor = backbone.preprocess(drawing_image())
    assert tensor.shape == (3, 16, 16)

    batch = torch.stack([tensor, tensor])
    features = backbone.encode(batch)
    assert features.shape == (2, 128)
    norms = features.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    assert backbone.encode_text(["a"]) is None


def test_fake_backbone_is_deterministic() -> None:
    from _imagegen import photo_image

    a = FakeBackbone()
    b = FakeBackbone()
    image = photo_image()
    fa = a.encode(a.preprocess(image).unsqueeze(0))
    fb = b.encode(b.preprocess(image).unsqueeze(0))
    assert torch.allclose(fa, fb)


def test_unknown_backbone_raises() -> None:
    with pytest.raises(ValueError, match="Unknown backbone"):
        build_backbone("nope", "cpu")


def test_registry_model_names_exist() -> None:
    """VERIFY the pinned model/pretrained names still exist (no download)."""
    import open_clip

    pretrained = open_clip.list_pretrained()
    assert ("ViT-SO400M-14-SigLIP", "webli") in pretrained
    assert ("ViT-L-14", "laion2b_s32b_b82k") in pretrained

    timm = pytest.importorskip("timm")
    names = set(timm.list_models())
    assert "vit_small_patch14_dinov2" in names
    assert "vit_base_patch14_dinov2" in names


def test_fake_features_separate_synthetic_classes() -> None:
    """Sanity check that the fake backbone is usable for pipeline tests."""
    from sklearn.linear_model import LogisticRegression

    from _imagegen import drawing_image, photo_image

    backbone = build_backbone("fake", "cpu")

    def features(image) -> np.ndarray:
        return backbone.encode(backbone.preprocess(image).unsqueeze(0))[0].numpy()

    rows, labels = [], []
    for i in range(30):
        rows.append(features(drawing_image(seed=i)))
        labels.append(0)
        rows.append(features(photo_image(seed=i)))
        labels.append(1)
    X = np.stack(rows)
    clf = LogisticRegression(max_iter=1000).fit(X, labels)
    assert clf.score(X, labels) > 0.95
