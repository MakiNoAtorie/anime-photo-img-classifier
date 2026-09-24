"""Adapter for the optional deepghs anime/real specialist model (G15, G26).

The package API is verified (V1): ``from imgutils.validate import
anime_real_score`` returns a mapping such as ``{'anime': p, 'real': q}``.
Everything here is import-lazy and optional so the tool works without the extra.
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

# Label names the classifier has used for the "real photo" class.
_REAL_KEYS = ("real", "photo", "photograph")

_INSTALL_HINT = (
    "dghs-imgutils is not installed. Install it with `pip install -e \".[deepghs]\"` "
    "or disable backbone.extra_features.deepghs_real."
)


def _import_score_fn():
    """Import the scoring function (the same import the scorer uses)."""
    from imgutils.validate import anime_real_score

    return anime_real_score


def deepghs_available() -> bool:
    """True if ``dghs-imgutils`` and its anime/real scorer are importable."""
    try:
        _import_score_fn()
    except Exception:
        return False
    return True


def require_deepghs() -> None:
    """Raise a clear ``RuntimeError`` if the optional dependency is missing."""
    if not deepghs_available():
        raise RuntimeError(_INSTALL_HINT)


def real_score(path: Path | str) -> float:
    """Return P(real) in [0, 1] from the deepghs classifier.

    Accepts either a mapping ``{'anime': p, 'real': q}`` or a bare score,
    depending on the installed version. An unrecognized mapping raises instead
    of guessing, because picking the wrong key would silently invert the score.
    """
    try:
        score_fn = _import_score_fn()
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(_INSTALL_HINT) from exc

    result = score_fn(str(path))
    if isinstance(result, dict):
        for key in _REAL_KEYS:
            if key in result:
                return float(result[key])
        raise RuntimeError(
            f"Unexpected deepghs result keys {sorted(result)}; expected one of {_REAL_KEYS}."
        )
    return float(result)


def run_baseline(config) -> Path:
    """Run the specialist model on the test set and write a report (M3)."""
    require_deepghs()
    from ..evaluate import run_specialist_baseline

    return run_specialist_baseline(config, "deepghs_anime_real", real_score)
