"""Sort/label a folder of new images, with an 'unsure' bucket (G22, §9.13).

Reuses the same scan -> features path so embeddings are cached. The output is
``dst/<class>/`` plus ``dst/unsure/`` and ``dst/broken/`` and a
``predictions.csv`` covering every input file (broken ones included).
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .imageio import ImageLoadError, load_rgb, sha1_file
from .models import load_any, load_latest_any, predict_proba_paths
from .predict_utils import unique_destination
from .scan import iter_images

log = logging.getLogger(__name__)


def _collect_inputs(config: Config, src: Path) -> pd.DataFrame:
    """Hash and integrity-check every image under ``src``."""
    extensions = set(config.image.extensions)
    rows = []
    for path in iter_images(src, extensions):
        row = {"path": str(path), "sha1": None, "status": "ok", "error": None}
        try:
            load_rgb(path, tuple(config.image.alpha_background), config.image.max_pixels)
            row["sha1"] = sha1_file(path)
        except ImageLoadError as exc:
            row["status"] = exc.reason
            row["error"] = exc.message
        except Exception as exc:  # noqa: BLE001
            row["status"] = "broken"
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    return pd.DataFrame(rows)


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
    if not src.exists():
        raise FileNotFoundError(f"Source folder does not exist: {src}")
    mode = mode or config.predict.mode
    threshold = float(threshold if threshold is not None else config.predict.unsure_threshold)
    if mode not in {"copy", "move", "csv_only"}:
        raise ValueError(f"Unknown mode {mode!r}; use copy, move or csv_only.")
    if mode == "move" and not yes:
        raise ValueError("`--mode move` deletes files from the source. Re-run with --yes to confirm.")

    bundle_path = Path(model) if model else load_latest_any(config)
    bundle = load_any(bundle_path)
    if set(bundle.classes) != set(config.classes):
        raise ValueError(
            f"Model classes {bundle.classes} != config classes {config.classes}."
        )

    frame = _collect_inputs(config, src)
    ok = frame[frame["status"] == "ok"].reset_index(drop=True)
    if not ok.empty:
        probs = predict_proba_paths(
            bundle,
            ok["path"].tolist(),
            config,
            shas=ok["sha1"].tolist(),
        )
        preds = probs.argmax(axis=1)
        confidence = probs.max(axis=1)
        ok = ok.assign(
            pred=[bundle.classes[i] for i in preds],
            confidence=confidence,
        )
        for i, cls in enumerate(bundle.classes):
            ok[f"p_{cls}"] = probs[:, i]

    # Assemble the CSV: every input file, broken ones with pred=broken.
    records = []
    for _, row in frame.iterrows():
        sha1 = row["sha1"] if isinstance(row["sha1"], str) else None
        record = {
            "path": row["path"],
            "sha1": sha1,
            "pred": "broken" if row["status"] != "ok" else None,
            "confidence": None,
            "dest": None,
        }
        for cls in config.classes:
            record[f"p_{cls}"] = None
        records.append(record)
    lookup = {row["path"]: row for _, row in ok.iterrows()}
    for record in records:
        row = lookup.get(record["path"])
        if row is None:
            continue
        record["pred"] = row["pred"]
        record["confidence"] = float(row["confidence"])
        for cls in config.classes:
            if f"p_{cls}" in row and pd.notna(row[f"p_{cls}"]):
                record[f"p_{cls}"] = float(row[f"p_{cls}"])

    dst.mkdir(parents=True, exist_ok=True)
    used: dict[str, Path] = {}

    if mode != "csv_only":
        # Create all buckets up front so they exist even when empty.
        for bucket in [*config.classes, config.skip_label, "unsure", "broken"]:
            (dst / bucket).mkdir(parents=True, exist_ok=True)
        for record in records:
            source = Path(record["path"])
            if record["pred"] == "broken":
                bucket = "broken"
            elif record["confidence"] is None:
                bucket = "unsure"
            elif record["confidence"] < threshold:
                bucket = "unsure"
            else:
                bucket = record["pred"]
            target = unique_destination(dst / bucket / source.name, record["sha1"] or "unknown", used)
            _transfer(source, target, mode)
            record["dest"] = str(target)

    csv_path = dst / "predictions.csv"
    pd.DataFrame(records).to_csv(csv_path, index=False)
    n_unsure = sum(1 for r in records if r.get("dest") and "unsure" in str(r["dest"]))
    log.info(
        "Predicted %d files (%d unsure, %d broken) -> %s",
        len(records),
        n_unsure,
        sum(1 for r in records if r["pred"] == "broken"),
        csv_path,
    )
    return csv_path


def _transfer(source: Path, target: Path, mode: str) -> None:
    if mode == "copy":
        shutil.copy2(source, target)
    elif mode == "move":
        shutil.move(str(source), str(target))
