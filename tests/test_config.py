"""Config loading and validation tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from pvd.config import Config, load_config, write_default_config


def test_defaults_when_file_missing(tmp_path: Path) -> None:
    cfg = load_config(tmp_path / "does_not_exist.yaml")
    assert cfg.classes == ["photo", "drawing"]
    assert cfg.skip_label == "skip"


def test_roundtrip_default_config(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    write_default_config(path)
    cfg = load_config(path)
    assert cfg == Config(project_root=cfg.project_root)


def test_skip_label_cannot_be_a_class() -> None:
    with pytest.raises(ValueError, match="skip_label"):
        Config(classes=["photo", "skip"], skip_label="skip")


def test_requires_two_classes() -> None:
    with pytest.raises(ValueError, match="at least two classes"):
        Config(classes=["photo"])


def test_duplicate_classes_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        Config(classes=["photo", "photo"])


def test_prompts_for_unknown_class_rejected() -> None:
    with pytest.raises(ValueError, match="non-classes"):
        Config(classes=["photo", "drawing"], zeroshot={"prompts": {"cat": ["a cat"]}})


def test_extra_key_rejected(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"classes": ["photo", "drawing"], "typo_field": 1}))
    with pytest.raises(ValueError):
        load_config(path)


def test_paths_resolve_relative_to_root(tmp_path: Path) -> None:
    cfg = Config(project_root=tmp_path)
    assert cfg.pool_dir == (tmp_path / "data/pool").resolve()
    assert cfg.manifest_path == (tmp_path / "cache/manifest.parquet").resolve()


def test_extensions_normalized() -> None:
    cfg = Config(image={"extensions": ["JPG", "png", ".WebP"]})
    assert cfg.image.extensions == [".jpg", ".png", ".webp"]
