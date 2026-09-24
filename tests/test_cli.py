"""CLI smoke tests: every command is registered and --help works."""

from __future__ import annotations

from pathlib import Path

import yaml
from typer.testing import CliRunner

from _imagegen import photo_image
from pvd.cli import app
from pvd.config import Config

runner = CliRunner()


def test_help_lists_all_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, result.output
    for command in (
        "scan",
        "embed",
        "zeroshot",
        "baseline-deepghs",
        "train",
        "eval",
        "review",
        "audit",
        "ablate",
        "predict",
        "finetune",
        "export-labels",
    ):
        assert command in result.stdout


def test_unknown_command_fails() -> None:
    result = runner.invoke(app, ["nope"])
    assert result.exit_code != 0


def test_scan_command_end_to_end(tmp_path: Path) -> None:
    for sub in ("data/pool", "data/seed", "data/test"):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)
    photo_image().save(tmp_path / "data" / "pool" / "a.png")

    payload = Config(project_root=tmp_path).to_dict()
    payload["project_root"] = str(tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    result = runner.invoke(app, ["--config", str(config_path), "scan"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "cache" / "manifest.parquet").exists()
