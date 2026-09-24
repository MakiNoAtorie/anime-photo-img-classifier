"""deepghs adapter handling without installing the optional package (V1)."""

from __future__ import annotations

import sys
import types


def _install_fake_imgutils(monkeypatch, score) -> None:
    package = types.ModuleType("imgutils")
    validate = types.ModuleType("imgutils.validate")
    real = types.ModuleType("imgutils.validate.real")
    validate.anime_real_score = lambda path, **kwargs: score
    real.anime_real_score = validate.anime_real_score
    monkeypatch.setitem(sys.modules, "imgutils", package)
    monkeypatch.setitem(sys.modules, "imgutils.validate", validate)
    monkeypatch.setitem(sys.modules, "imgutils.validate.real", real)


def test_real_score_reads_dict(monkeypatch) -> None:
    _install_fake_imgutils(monkeypatch, {"anime": 0.1, "real": 0.9})
    from pvd.backbones.deepghs import deepghs_available, real_score

    assert deepghs_available() is True
    assert real_score("whatever.png") == 0.9


def test_real_score_reads_bare_float(monkeypatch) -> None:
    _install_fake_imgutils(monkeypatch, 0.42)
    from pvd.backbones.deepghs import real_score

    assert real_score("whatever.png") == 0.42
