"""Shared test fixtures.

Unit tests must never download models or depend on the dev machine. Image
generators and the ``fake`` backbone are defined here / in later milestones so
the whole pipeline can be exercised hermetically.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pvd.config import Config, write_default_config


@pytest.fixture()
def tmp_project(tmp_path: Path) -> Config:
    """Create an isolated project tree with a written config."""
    write_default_config(tmp_path / "config.yaml")
    for sub in ("data/pool", "data/seed", "data/test"):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)
    return Config(project_root=tmp_path)
