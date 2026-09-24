"""Active-learning batch sampler (G19, G16, §9.9).

Each batch mixes hard examples (small top-2 margin) with a random slice. The
random slice is what makes the reported error rate on a batch unbiased: the
uncertain part is deliberately hard and will look bad even when the model
improves (G12). A diversity filter keeps near-identical uncertain images out of
the same batch so a reviewer sees variety.
"""

from __future__ import annotations

import logging

import numpy as np

from .config import Config

log = logging.getLogger(__name__)

DIVERSITY_COSINE = 0.95


def _top2_margin(probs: np.ndarray) -> np.ndarray:
    """Distance between the best and second-best class probability.

    Smaller means more uncertain. Works for any number of classes.
    """
    if probs.shape[1] < 2:
        return np.ones(len(probs), dtype=np.float64)
    sorted_probs = np.sort(probs, axis=1)
    return sorted_probs[:, -1] - sorted_probs[:, -2]


def next_batch(
    probs: np.ndarray,
    candidate_indices: list[int],
    features: np.ndarray,
    config: Config,
    rng: np.random.Generator,
) -> list[tuple[int, str]]:
    """Return ``[(dataset_index, 'uncertain'|'random'), ...]`` for one batch.

    ``probs`` and ``features`` are indexed by the caller's global index space;
    ``candidate_indices`` lists which of those rows are still unlabeled.
    ``features`` should be the unit-norm global block: the diversity filter
    compares dot products against a cosine threshold.
    """
    batch_size = min(config.review.batch_size, len(candidate_indices))
    if batch_size == 0:
        return []

    candidates = list(candidate_indices)
    n_uncertain = int(round(batch_size * (1.0 - config.review.random_fraction)))
    n_random = batch_size - n_uncertain

    # Rank by uncertainty (most uncertain first).
    margins = _top2_margin(probs[candidates])
    order = np.argsort(margins, kind="stable")
    ranked = [candidates[i] for i in order]

    use_diversity = config.review.diversity and features is not None and features.size > 0

    selected: list[int] = []
    picked_unit: list[np.ndarray] = []  # unit vectors of the picks, normalized lazily
    for index in ranked:
        if len(selected) >= n_uncertain:
            break
        if use_diversity:
            vector = features[index] / max(float(np.linalg.norm(features[index])), 1e-12)
            if picked_unit and float(np.max(np.stack(picked_unit) @ vector)) > DIVERSITY_COSINE:
                continue
            picked_unit.append(vector)
        selected.append(index)

    uncertain = [(index, "uncertain") for index in selected]

    chosen = set(selected)
    remaining = [index for index in candidates if index not in chosen]
    rng.shuffle(remaining)
    # If the diversity filter left the uncertain slice short (many near-
    # duplicates), top up with random picks so the batch keeps its size.
    n_random += n_uncertain - len(selected)
    random_picks = [(index, "random") for index in remaining[:n_random]]

    batch = uncertain + random_picks
    # Shuffle so the reviewer cannot infer which items are the hard ones (G19).
    permutation = rng.permutation(len(batch))
    return [batch[i] for i in permutation]
