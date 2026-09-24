"""Batch sampler behaviour (M5)."""

from __future__ import annotations

import numpy as np

from pvd.config import Config
from pvd.sampler import next_batch


def _probs(n: int) -> np.ndarray:
    p = np.linspace(0.5, 1.0, n)
    return np.column_stack([p, 1 - p])


def test_batch_composition_and_determinism() -> None:
    cfg = Config(review={"batch_size": 10, "random_fraction": 0.4, "diversity": False})
    probs = _probs(20)
    candidates = list(range(20))
    features = np.eye(20, dtype=np.float32)

    first = next_batch(probs, candidates, features, cfg, np.random.default_rng(0))
    assert len(first) == 10
    kinds = [kind for _, kind in first]
    assert kinds.count("uncertain") == 6
    assert kinds.count("random") == 4

    second = next_batch(probs, candidates, features, cfg, np.random.default_rng(0))
    assert first == second


def test_uncertain_items_have_small_margin() -> None:
    cfg = Config(review={"batch_size": 4, "random_fraction": 0.0, "diversity": False})
    probs = _probs(10)
    batch = next_batch(probs, list(range(10)), np.eye(10, dtype=np.float32), cfg, np.random.default_rng(1))
    # Margin is smallest at index 0 (p=0.5 -> margin 0) and grows with index.
    assert {index for index, _ in batch} == {0, 1, 2, 3}


def test_diversity_filters_near_duplicates() -> None:
    cfg = Config(review={"batch_size": 5, "random_fraction": 0.0, "diversity": True})
    probs = _probs(10)
    features = np.ones((10, 3), dtype=np.float32)  # all identical
    batch = next_batch(probs, list(range(10)), features, cfg, np.random.default_rng(2))
    assert sum(1 for _, kind in batch if kind == "uncertain") == 1


def test_multi_class_margin() -> None:
    cfg = Config(review={"batch_size": 2, "random_fraction": 0.0, "diversity": False})
    probs = np.array([[0.5, 0.3, 0.2], [0.1, 0.8, 0.1], [0.34, 0.33, 0.33]])
    batch = next_batch(probs, [0, 1, 2], np.eye(3, dtype=np.float32), cfg, np.random.default_rng(3))
    picked = {index for index, _ in batch}
    assert 2 in picked  # the genuinely ambiguous row


def test_empty_candidates() -> None:
    cfg = Config()
    batch = next_batch(np.zeros((0, 2)), [], np.zeros((0, 2), np.float32), cfg, np.random.default_rng(0))
    assert batch == []
