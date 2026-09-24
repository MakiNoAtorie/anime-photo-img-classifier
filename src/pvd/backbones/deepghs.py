"""Adapter for the optional deepghs anime/real specialist model (G15, G26).

The package API is verified (V1): ``imgutils.validate.real.anime_real_score``.
Everything here is import-lazy and optional so the tool works without the extra.
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

_REAL_KEYS = ("real", "photo", "photograph")


def deepghs_available() -> bool:
    """True if ``dghs-imgutils`` is importable."""
    try:
        import imgutils.validate.real  # noqa: F401
    except Exception:
        return False
    return True


def real_score(path: Path | str) -> float:
    """Return P(real) in [0, 1] from the deepghs classifier.

    The public function may return either a mapping ``{'anime': p, 'real': q}``
    or a bare score depending on the installed version, so both are handled.
    Raises ``RuntimeError`` if the optional dependency is missing.
    """
    try:
        from imgutils.validate import anime_real_score
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "dghs-imgutils is not installed. Install it with `pip install dghs-imgutils` "
            "or disable backbone.extra_features.deepghs_real."
        ) from exc

    result = anime_real_score(str(path))
    if isinstance(result, dict):
        for key in _REAL_KEYS:
            if key in result:
                return float(result[key])
        # Unknown key naming: pick the maximum-probability label as a fallback.
        return float(max(result.values()))
    return float(result)


def run_baseline(config) -> Path:
    """Run the specialist model on the test set and write a report (M3)."""
    if not deepghs_available():
        raise RuntimeError(
            "dghs-imgutils is not installed; `pvd baseline-deepghs` needs it. "
            "Install with `pip install dghs-imgutils`."
        )
    from ..evaluate import run_specialist_baseline

    return run_specialist_baseline(config, "deepghs_anime_real", real_score)
