"""Sort/label a folder of new images, with an 'unsure' bucket (G22, §9.13).

Reuses the same features path as training, so embeddings are cached. The output
is ``dst/<class>/`` plus ``dst/unsure/`` and ``dst/broken/``, and a
``predictions.csv`` with one row per input file (broken ones included, with
``pred=broken``).

Safety rules:

- files are never overwritten: a name clash gets ``_<sha1[:8]>`` appended;
- the CSV is written **before** any file is copied or moved, and rewritten
  after, so an interrupted ``--mode move`` still leaves a record of where every
  file was going;
- ``--mode move`` requires ``--yes``.
"""

from __future__ import annotations

import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .imageio import ImageLoadError, probe_image, sha1_file
from .models import load_any, newest_bundle_path, predict_proba_paths
from .scan import iter_images

log = logging.getLogger(__name__)

UNSURE, BROKEN = "unsure", "broken"


def unique_destination(target: Path, sha1: str | None, reserved: set[Path] | None = None) -> Path:
    """Return a collision-free path; never overwrite an existing file (§9.13).

    ``reserved`` holds destinations already planned in this run, so two inputs
    with the same file name (from different subfolders) don't collide either.
    On a clash the first 8 characters of the content hash are appended.
    """
    reserved = reserved if reserved is not None else set()
    tag = sha1[:8] if isinstance(sha1, str) and sha1 else "unknown"
    if not target.exists() and target not in reserved:
        return target
    stem, suffix = target.stem, target.suffix
    candidate = target.with_name(f"{stem}_{tag}{suffix}")
    counter = 2
    while candidate.exists() or candidate in reserved:
        candidate = target.with_name(f"{stem}_{tag}_{counter}{suffix}")
        counter += 1
    return candidate


def _collect_inputs(config: Config, src: Path) -> pd.DataFrame:
    """Hash and header-check every image under ``src`` (full decode happens once, later)."""
    extensions = set(config.image.extensions)
    rows = []
    for path in iter_images(src, extensions, follow_symlinks=False):
        row = {"abs_path": str(path), "sha1": None, "status": "ok", "error": None}
        try:
            row["sha1"] = sha1_file(path)
            info = probe_image(path)
            if config.image.max_pixels and info.width * info.height > config.image.max_pixels:
                raise ImageLoadError("too_large", f"{info.width}x{info.height} exceeds max_pixels")
        except ImageLoadError as exc:
            row["status"] = exc.reason
            row["error"] = str(exc)
        except OSError as exc:
            row["status"] = BROKEN
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    return pd.DataFrame(rows, columns=["abs_path", "sha1", "status", "error"])


def _csv_path(dst: Path) -> Path:
    """``predictions.csv``, or a timestamped name if a previous run's CSV exists."""
    path = dst / "predictions.csv"
    if not path.exists():
        return path
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return unique_destination(dst / f"predictions_{stamp}.csv", None)


def run_predict(
    config: Config,
    src: Path,
    dst: Path,
    model: Path | None = None,
    threshold: float | None = None,
    mode: str | None = None,
    yes: bool = False,
) -> Path:
    """Predict and optionally sort a folder; return the predictions CSV path."""
    src, dst = Path(src), Path(dst)
    if not src.is_dir():
        raise FileNotFoundError(f"Source folder does not exist: {src}")
    mode = mode or config.predict.mode
    threshold = float(threshold if threshold is not None else config.predict.unsure_threshold)
    if mode not in {"copy", "move", "csv_only"}:
        raise ValueError(f"Unknown mode {mode!r}; use copy, move or csv_only.")
    if mode == "move" and not yes:
        raise ValueError("`--mode move` deletes files from the source. Re-run with --yes to confirm.")

    bundle_path = Path(model) if model else newest_bundle_path(config)
    bundle = load_any(bundle_path)
    if set(bundle.classes) != set(config.classes):
        raise ValueError(f"Model classes {bundle.classes} != config classes {config.classes}.")
    classes = list(bundle.classes)

    frame = _collect_inputs(config, src)
    if frame.empty:
        log.warning("No images with extensions %s under %s.", config.image.extensions, src)

    ok_mask = (frame["status"] == "ok").to_numpy()
    probs = np.full((len(frame), len(classes)), np.nan)
    if ok_mask.any():
        ok_probs, valid = predict_proba_paths(
            bundle,
            frame.loc[ok_mask, "abs_path"].tolist(),
            config,
            shas=frame.loc[ok_mask, "sha1"].tolist(),
        )
        probs[ok_mask] = ok_probs
        # Images that passed the header check but failed to decode are broken too.
        failed_rows = np.flatnonzero(ok_mask)[~valid]
        frame.loc[failed_rows, "status"] = BROKEN
        frame.loc[failed_rows, "error"] = "could not be decoded/embedded"

    records = []
    for i, row in frame.iterrows():
        broken = row["status"] != "ok"
        record = {
            "path": Path(row["abs_path"]).relative_to(src).as_posix(),
            "sha1": row["sha1"],
            "pred": BROKEN if broken else classes[int(np.argmax(probs[i]))],
            "confidence": None if broken else float(np.max(probs[i])),
        }
        for c, name in enumerate(classes):
            record[f"p_{name}"] = None if broken else float(probs[i, c])
        record["bucket"] = (
            BROKEN if broken else (record["pred"] if record["confidence"] >= threshold else UNSURE)
        )
        record["dest"] = None
        record["error"] = row["error"]
        records.append(record)

    dst.mkdir(parents=True, exist_ok=True)
    csv_path = _csv_path(dst)

    if mode != "csv_only":
        # Create all buckets up front so they exist even when empty.
        for bucket in [*classes, UNSURE, BROKEN]:
            (dst / bucket).mkdir(parents=True, exist_ok=True)
        reserved: set[Path] = set()
        for record in records:
            target = unique_destination(dst / record["bucket"] / Path(record["path"]).name, record["sha1"], reserved)
            reserved.add(target)
            record["dest"] = target.relative_to(dst).as_posix()

    columns = ["path", "sha1", "pred", "confidence", *[f"p_{c}" for c in classes], "bucket", "dest", "error"]
    # Record the plan first: if a move is interrupted, the CSV still says where
    # every file was going.
    pd.DataFrame(records, columns=columns).to_csv(csv_path, index=False)

    if mode != "csv_only":
        failures = 0
        for record in records:
            try:
                _transfer(src / record["path"], dst / record["dest"], mode)
            except OSError as exc:
                failures += 1
                record["error"] = f"transfer failed: {exc}"
                log.error("Could not %s %s: %s", mode, record["path"], exc)
        pd.DataFrame(records, columns=columns).to_csv(csv_path, index=False)
        if failures:
            log.warning("%d file(s) could not be transferred; see the error column.", failures)

    log.info(
        "Predicted %d files (%d unsure, %d broken) -> %s",
        len(records),
        sum(1 for r in records if r["bucket"] == UNSURE),
        sum(1 for r in records if r["bucket"] == BROKEN),
        csv_path,
    )
    return csv_path


def _transfer(source: Path, target: Path, mode: str) -> None:
    """Copy or move one file; ``copy2`` keeps timestamps."""
    if mode == "copy":
        shutil.copy2(source, target)
    elif mode == "move":
        shutil.move(str(source), str(target))
