"""Device selection: cuda -> mps -> cpu (G10).

``torch`` is imported lazily so that lightweight commands (e.g. ``pvd scan``)
and the config layer keep working before the heavy dependencies are installed.
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)

# MPS lacks a few ops; PyTorch can fall back to CPU for them if this is set.
# We document the env var in the README instead of setting it silently.
MPS_FALLBACK_ENV = "PYTORCH_ENABLE_MPS_FALLBACK"


_logged: set[str] = set()


def _log_once(message: str, *args) -> None:
    """Log a device message only the first time it occurs in this process."""
    text = message % args
    if text not in _logged:
        _logged.add(text)
        log.info(text)


def pick_device(pref: str = "auto") -> str:
    """Return ``'cuda'``, ``'mps'`` or ``'cpu'``.

    ``pref='auto'`` tries CUDA, then Apple MPS, then CPU. An explicit preference
    is honoured when available, otherwise we fall back with a warning so a
    config copied between machines still runs.
    """
    import torch  # local import: see module docstring

    available = {
        "cuda": bool(torch.cuda.is_available()),
        "mps": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        "cpu": True,
    }

    if pref == "auto":
        for candidate in ("cuda", "mps", "cpu"):
            if available[candidate]:
                _log_once("Using device: %s", candidate)
                return candidate
        return "cpu"

    if pref in available and available[pref]:
        _log_once("Using device: %s", pref)
        return pref

    log.warning("Requested device %r unavailable (available=%s); falling back to cpu", pref, available)
    return "cpu"


def describe_environment() -> dict[str, object]:
    """Small diagnostic dict for reports (no side effects)."""
    import torch

    return {
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "mps_available": bool(
            getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        ),
        f"{MPS_FALLBACK_ENV}": os.environ.get(MPS_FALLBACK_ENV),
    }
