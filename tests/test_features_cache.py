"""Embedding cache behaviour tests (M2 acceptance).

Acceptance: second embed run computes 0 new embeddings; renaming a file computes
0 new; deleting the cache recomputes all; interrupted run resumes; ``num_workers``
0 and >0 both work.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from _imagegen import drawing_image, photo_image
from pvd.config import Config
from pvd.features import (
    embedding_cache_path,
    feature_key,
    get_features,
    load_embedding_cache,
    run_embed,
)
from pvd.scan import load_manifest, run_scan


def _populate(cfg: Config, n: int = 3) -> None:
    (cfg.seed_dir / "photo").mkdir(parents=True, exist_ok=True)
    (cfg.seed_dir / "drawing").mkdir(parents=True, exist_ok=True)
    for i in range(n):
        photo_image(seed=i).save(cfg.seed_dir / "photo" / f"p{i}.png")
        drawing_image(seed=i).save(cfg.seed_dir / "drawing" / f"d{i}.png")


def _fake_config(tmp_project: Config) -> Config:
    tmp_project.backbone.id = "fake"
    tmp_project.num_workers = 0
    return tmp_project


def test_feature_key_reflects_recipe(tmp_project: Config) -> None:
    cfg = _fake_config(tmp_project)
    assert feature_key("clip_b32", cfg) == "clip_b32"
    cfg.backbone.tiles.enabled = True
    cfg.backbone.tiles.count = 5
    assert feature_key("clip_b32", cfg) == "clip_b32+tiles5"
    cfg.backbone.tiles.enabled = False
    cfg.backbone.extra_features.deepghs_real = True
    assert feature_key("clip_b32", cfg) == "clip_b32+deepghs"


def test_embed_creates_cache_and_normalizes(tmp_project: Config) -> None:
    cfg = _fake_config(tmp_project)
    _populate(cfg)
    run_scan(cfg)
    path = run_embed(cfg)
    assert path.exists()

    key = feature_key("fake", cfg)
    mapping, meta = load_embedding_cache(cfg, key)
    assert len(mapping) == 6
    assert meta["block_sizes"]["global"] == 128
    assert meta["normalization"] == "l2"
    sample = next(iter(mapping.values()))
    assert sample.shape == (128,)
    assert np.isclose(np.linalg.norm(sample), 1.0, atol=1e-5)


def test_second_run_reuses_cache_without_reading_files(tmp_project: Config) -> None:
    cfg = _fake_config(tmp_project)
    _populate(cfg)
    run_scan(cfg)
    run_embed(cfg)

    manifest = load_manifest(cfg)
    shas = manifest["sha1"].tolist()
    paths = [str(cfg.resolve(p)) for p in manifest["path"].tolist()]
    first = get_features(cfg, shas, paths, "fake")

    # Delete the source files: a cache hit must still produce the same vectors.
    for path in cfg.seed_dir.rglob("*.png"):
        path.unlink()
    second = get_features(cfg, shas, paths, "fake")
    assert np.allclose(first, second)
    assert not np.allclose(first, 0)


def test_deleting_cache_forces_recompute(tmp_project: Config) -> None:
    cfg = _fake_config(tmp_project)
    _populate(cfg)
    run_scan(cfg)
    key = feature_key("fake", cfg)
    run_embed(cfg)
    embedding_cache_path(cfg, key).unlink()

    manifest = load_manifest(cfg)
    shas = manifest["sha1"].tolist()
    paths = [str(cfg.resolve(p)) for p in manifest["path"].tolist()]
    # Recompute now needs the files, which we deleted, so vectors are zero.
    for path in cfg.seed_dir.rglob("*.png"):
        path.unlink()
    zeros = get_features(cfg, shas, paths, "fake")
    assert np.allclose(zeros, 0)

    # Restore files -> recompute works again and cache is repopulated.
    _populate(cfg)
    recomputed = get_features(cfg, shas, paths, "fake")
    assert not np.allclose(recomputed, 0)
    assert embedding_cache_path(cfg, key).exists()


def test_rename_does_not_recompute(tmp_project: Config, monkeypatch) -> None:
    cfg = _fake_config(tmp_project)
    _populate(cfg, n=1)
    run_scan(cfg)
    key = feature_key("fake", cfg)
    run_embed(cfg)
    before, _ = load_embedding_cache(cfg, key)

    source = cfg.seed_dir / "photo" / "p0.png"
    renamed = cfg.seed_dir / "photo" / "p0_renamed.png"
    source.rename(renamed)
    run_scan(cfg)

    # Nothing may be recomputed: building the backbone at all is a failure.
    import pvd.features as features

    def no_backbone(*args, **kwargs):
        raise AssertionError("rename triggered recomputation")

    monkeypatch.setattr(features, "build_backbone", no_backbone)
    run_embed(cfg)
    after, _ = load_embedding_cache(cfg, key)

    assert set(before) == set(after)
    assert len(after) == len(before)


@pytest.mark.parametrize("workers", [0, 2])
def test_num_workers_both_work(tmp_project: Config, workers: int) -> None:
    cfg = _fake_config(tmp_project)
    cfg.num_workers = workers
    _populate(cfg)
    run_scan(cfg)
    run_embed(cfg)
    mapping, _ = load_embedding_cache(cfg, feature_key("fake", cfg))
    assert len(mapping) == 6


def test_partial_cache_is_completed(tmp_project: Config) -> None:
    """Simulate an interrupted run: only the missing vectors are recomputed."""
    cfg = _fake_config(tmp_project)
    _populate(cfg)
    run_scan(cfg)
    key = feature_key("fake", cfg)
    run_embed(cfg)
    full, meta = load_embedding_cache(cfg, key)
    shas = list(full.keys())

    # Keep only the first half, as if the process had died mid-run.
    partial = {sha: full[sha] for sha in shas[: len(shas) // 2]}
    cfg.embeddings_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        embedding_cache_path(cfg, key),
        sha1=np.array(list(partial), dtype=str),
        feats=np.stack(list(partial.values())).astype(np.float32),
        meta=np.array(json.dumps(meta)),
    )

    manifest = load_manifest(cfg)
    order = manifest["sha1"].tolist()
    paths = [str(cfg.resolve(p)) for p in manifest["path"].tolist()]
    matrix = get_features(cfg, order, paths, "fake")

    completed, _ = load_embedding_cache(cfg, key)
    assert set(completed) == set(full)
    for i, sha in enumerate(order):
        # Batch-size-dependent BLAS rounding can differ at ~1e-7; that is fine.
        assert np.allclose(matrix[i], full[sha], atol=1e-5)

