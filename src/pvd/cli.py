"""``pvd`` command-line entry point.

All commands are registered here so ``pvd --help`` is a complete map of the
tool. Heavy imports happen inside the command bodies (or inside ``main``) so
that ``--help`` and ``pvd scan`` work without loading torch (G9).

Every command is wrapped so a user-facing error (missing test set, unknown
backbone, absent optional dependency) prints a single clear line and exits 1
instead of dumping a traceback.
"""

from __future__ import annotations

import functools
import logging
import sys
from pathlib import Path

import typer

from .config import Config, load_config

log = logging.getLogger(__name__)

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Photo-vs-drawing classifier with a human-in-the-loop correction loop.",
)

# Exceptions that represent "the user can fix this", not a bug.
_USER_ERRORS = (ValueError, FileNotFoundError, RuntimeError, NotImplementedError)


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _guard(func):
    """Turn expected errors into a clean message + exit code 1."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except _USER_ERRORS as exc:
            typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from exc

    wrapper.__annotations__ = getattr(func, "__annotations__", {})
    return wrapper


def _command(*args, **kwargs):
    """``@app.command`` plus the error guard, preserving the signature for Typer."""

    def decorator(func):
        return app.command(*args, **kwargs)(_guard(func))

    return decorator


@app.callback()
def _root(
    ctx: typer.Context,
    config: Path = typer.Option(
        Path("config.yaml"),
        "--config",
        "-c",
        help="Path to config.yaml (defaults are used if it does not exist).",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """Load configuration once and share it with every subcommand."""
    _setup_logging(verbose)
    ctx.obj = load_config(config)


def _cfg(ctx: typer.Context) -> Config:
    assert isinstance(ctx.obj, Config)
    return ctx.obj


# --------------------------------------------------------------------------- #
# Data preparation
# --------------------------------------------------------------------------- #
@_command()
def scan(ctx: typer.Context) -> None:
    """Build/refresh the image manifest and report broken files and duplicates."""
    from .scan import run_scan

    run_scan(_cfg(ctx))


@_command()
def embed(
    ctx: typer.Context,
    backbone: str = typer.Option(None, "--backbone", help="Override the configured backbone id."),
) -> None:
    """Compute and cache embeddings for every valid image."""
    from .features import run_embed

    run_embed(_cfg(ctx), backbone_id=backbone)


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #
@_command()
def zeroshot(ctx: typer.Context) -> None:
    """Zero-shot (CLIP/SigLIP text prompts) predictions and test-set report."""
    from .zeroshot import run_zeroshot

    run_zeroshot(_cfg(ctx))


@_command("baseline-deepghs")
def baseline_deepghs(ctx: typer.Context) -> None:
    """Run the optional deepghs anime/real model on the test set."""
    from .backbones.deepghs import run_baseline

    run_baseline(_cfg(ctx))


# --------------------------------------------------------------------------- #
# Probe / training
# --------------------------------------------------------------------------- #
@_command()
def train(ctx: typer.Context) -> None:
    """Train a CV-tuned linear probe on the current labels and save a bundle."""
    from .probe import run_train

    run_train(_cfg(ctx))


@_command()
def eval(
    ctx: typer.Context,
    model: Path = typer.Option(None, "--model", help="Model bundle (defaults to newest probe)."),
) -> None:
    """Full evaluation report of a saved model on the held-out test set."""
    from .evaluate import run_eval

    run_eval(_cfg(ctx), model=model)


# --------------------------------------------------------------------------- #
# Human-in-the-loop
# --------------------------------------------------------------------------- #
@_command()
def review(ctx: typer.Context) -> None:
    """Launch the Gradio active-learning review app."""
    from .review_app import run_review

    run_review(_cfg(ctx))


@_command()
def audit(ctx: typer.Context) -> None:
    """Report likely label mistakes, near-duplicate leakage and shortcut features."""
    from .audit import run_audit

    run_audit(_cfg(ctx))


@_command()
def ablate(ctx: typer.Context) -> None:
    """Compare backbones and feature blocks; write an ablation report."""
    from .ablation import run_ablation

    run_ablation(_cfg(ctx))


# --------------------------------------------------------------------------- #
# Inference / export
# --------------------------------------------------------------------------- #
@_command()
def predict(
    ctx: typer.Context,
    src: Path = typer.Option(..., "--src", help="Folder of new images to sort."),
    dst: Path = typer.Option(..., "--dst", help="Output folder."),
    model: Path = typer.Option(None, "--model", help="Model bundle; defaults to newest probe."),
    threshold: float = typer.Option(None, "--threshold", help="Max-probability cutoff for 'unsure'."),
    mode: str = typer.Option(None, "--mode", help="copy | move | csv_only"),
    yes: bool = typer.Option(False, "--yes", help="Required to allow --mode move."),
) -> None:
    """Sort/label a folder of images, with an 'unsure' bucket."""
    from .predict import run_predict

    run_predict(_cfg(ctx), src=src, dst=dst, model=model, threshold=threshold, mode=mode, yes=yes)


@_command()
def finetune(ctx: typer.Context) -> None:
    """Fine-tune a timm model on the collected labels (optional stage)."""
    from .finetune import run_finetune

    run_finetune(_cfg(ctx))


@_command("export-labels")
def export_labels(
    ctx: typer.Context,
    fmt: str = typer.Option("folders", "--format", help="csv | folders"),
    out: Path = typer.Option(Path("export"), "--out", help="Output location."),
) -> None:
    """Export the current labels as a CSV or as one folder per class."""
    from .labels import run_export

    run_export(_cfg(ctx), fmt=fmt, out=out)


def main() -> None:
    """Console-script entry point."""
    try:
        app()
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        sys.exit(130)


if __name__ == "__main__":  # pragma: no cover
    main()
