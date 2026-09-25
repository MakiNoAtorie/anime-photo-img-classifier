"""Device selection: cuda -> mps -> cpu (G10).

``torch`` is imported lazily so that lightweight commands (e.g. ``pvd scan``)
and the config layer keep working before the heavy dependencies are installed.
"""

from __future__ import annotations

import functools
import logging

log = logging.getLogger(__name__)


@functools.cache  # decide (and log) once per process
def pick_device(pref: str = "auto") -> str:
    """Return ``'cuda'``, ``'mps'`` or ``'cpu'``.

    ``pref='auto'`` tries CUDA, then Apple MPS, then CPU. An explicit preference
    is honoured when available, otherwise we fall back with a warning so a
    config copied between machines still runs.
    """
    import torch  # local import: see module docstring

    candidates = (("cuda", torch.cuda.is_available()), ("mps", torch.backends.mps.is_available()))
    available = [name for name, ok in candidates if ok] + ["cpu"]
    if pref == "auto" or pref in available:
        device = available[0] if pref == "auto" else pref
        log.info("Using device: %s", device)
        return device
    log.warning("Requested device %r unavailable (available=%s); falling back to cpu", pref, available)
    return "cpu"
