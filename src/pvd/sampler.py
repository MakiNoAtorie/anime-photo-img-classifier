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

    selected: list[int] = []
    for index in ranked:
        if len(selected) >= n_uncertain:
            break
        if config.review.diversity and features is not None and selected:
            similarity = features[selected] @ features[index]
            if float(np.max(similarity)) > DIVERSITY_COSINE:
                continue
        selected.append(index)

    uncertain = [(index, "uncertain") for index in selected]

    remaining = [index for index in candidates if index not in set(selected)]
    rng.shuffle(remaining)
    random_picks = [(index, "random") for index in remaining[:n_random]]

    batch = uncertain + random_picks
    # Shuffle so the reviewer cannot infer which items are the hard ones (G19).
    permutation = rng.permutation(len(batch))
    return [batch[i] for i in permutation]
