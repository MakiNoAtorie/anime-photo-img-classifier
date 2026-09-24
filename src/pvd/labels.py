"""Append-only label log with undo (G1, G17, §8.3).

Seed-folder labels are the source of truth for the seed split and are *not*
written here. The log records human decisions (``review``/``audit_fix``) and the
seed overrides discovered during audit. State is reconstructed on load by
replaying events and ignoring any batch that a later ``undo`` event reverts, so
the file can only ever grow and is crash-safe.
"""

from __future__ import annotations

import csv
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import Config

log = logging.getLogger(__name__)

VALID_SOURCES = {"seed_import", "review", "audit_fix", "undo"}
VALID_SAMPLED_AS = {"uncertain", "random", "audit", None}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class LabelEvent:
    """One line in ``labels/events.jsonl``."""

    sha1: str
    label: str
    path: str | None = None
    prev: str | None = None
    source: str = "review"
    batch_id: str | None = None
    model_pred: str | None = None
    model_conf: float | None = None
    sampled_as: str | None = None
    undo_batch_id: str | None = None
    ts: str = field(default_factory=_now)

    def to_json(self) -> str:
        payload = {k: v for k, v in asdict(self).items() if v is not None}
        return json.dumps(payload, sort_keys=True)


class LabelStore:
    """Loads and appends to the label event log."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._events: list[LabelEvent] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with open(self.path, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    log.warning("Skipping malformed label event at line %d", line_number)
                    continue
                allowed = {k: payload.get(k) for k in LabelEvent.__dataclass_fields__}
                allowed["sha1"] = allowed.get("sha1") or ""
                allowed["label"] = allowed.get("label") or ""
                if allowed.get("source") not in VALID_SOURCES:
                    allowed["source"] = "review"
                self._events.append(LabelEvent(**allowed))

    # -- state --------------------------------------------------------------
    def history(self) -> list[LabelEvent]:
        return list(self._events)

    def _undone_batches(self) -> set[str]:
        return {
            e.undo_batch_id
            for e in self._events
            if e.source == "undo" and e.undo_batch_id
        }

    def current(self) -> dict[str, str]:
        """Return ``sha1 -> label`` after applying overrides and undos."""
        undone = self._undone_batches()
        state: dict[str, str] = {}
        for event in self._events:
            if event.source == "undo" or not event.sha1 or not event.label:
                continue
            if event.batch_id and event.batch_id in undone:
                continue
            state[event.sha1] = event.label
        return state

    def next_batch_id(self) -> str:
        numbers = [
            int(e.batch_id[1:])
            for e in self._events
            if e.batch_id and e.batch_id.startswith("b") and e.batch_id[1:].isdigit()
        ]
        return f"b{(max(numbers) + 1) if numbers else 1:04d}"

    # -- writes -------------------------------------------------------------
    def _append(self, event: LabelEvent) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(event.to_json() + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._events.append(event)

    def add_batch(self, entries: list[LabelEvent]) -> str:
        """Append a batch of labels and return its batch id."""
        batch_id = self.next_batch_id()
        for event in entries:
            event.batch_id = batch_id
            if event.source not in VALID_SOURCES:
                event.source = "review"
            if event.sampled_as not in VALID_SAMPLED_AS:
                event.sampled_as = None
            self._append(event)
        return batch_id

    def undo_last(self) -> str | None:
        """Revert the most recent non-undone batch; return its id (or None)."""
        undone = self._undone_batches()
        last_batch: str | None = None
        for event in self._events:
            if event.source == "undo" or not event.batch_id:
                continue
            if event.batch_id in undone:
                continue
            last_batch = event.batch_id
        if last_batch is None:
            return None
        self._append(LabelEvent(sha1="", label="", source="undo", undo_batch_id=last_batch))
        return last_batch


# --------------------------------------------------------------------------- #
# Effective training labels
# --------------------------------------------------------------------------- #
def effective_labels(config: Config, manifest: pd.DataFrame) -> pd.DataFrame:
    """Return ``path, sha1, label`` rows safe to train on.

    Combines seed-folder labels with log overrides (``review``/``audit_fix``),
    drops ``skip`` and excluded statuses, and hard-excludes the test split so a
    test image can never leak into training (G11).
    """
    store = LabelStore(config.labels_file)
    overrides = store.current()

    rows: list[dict] = []
    for _, row in manifest.iterrows():
        if row["status"] != "ok":
            continue
        sha = row["sha1"]
        if sha in overrides:
            label = overrides[sha]
        elif row["split"] == "seed":
            label = row["folder_label"]
        else:
            # Pool images only enter training once a human labels them.
            continue
        if label is None or label == config.skip_label:
            continue
        if label not in config.classes:
            continue
        if row["split"] == "test":
            continue
        rows.append({"path": row["path"], "sha1": sha, "label": label})

    frame = pd.DataFrame(rows, columns=["path", "sha1", "label"])
    test_shas = set(manifest.loc[manifest["split"] == "test", "sha1"])
    leaked = set(frame["sha1"]) & test_shas
    if leaked:
        raise RuntimeError(
            f"Refusing to train: {len(leaked)} test images have labels. This is a bug."
        )
    return frame


def label_snapshot_sha(labels: pd.DataFrame) -> str:
    """Stable hash of a label set, stored in model bundles for reproducibility."""
    import hashlib

    digest = hashlib.sha1()
    for sha, label in sorted(zip(labels["sha1"], labels["label"])):
        digest.update(f"{sha}:{label}\n".encode())
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
def run_export(config: Config, fmt: str = "folders", out: Path = Path("export")) -> Path:
    """Export current labels as a CSV or as one folder per class."""
    from .scan import load_manifest

    manifest = load_manifest(config)
    labels = effective_labels(config, manifest)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    if fmt == "csv":
        path = out / "labels.csv"
        labels.to_csv(path, index=False)
        log.info("Exported %d labels -> %s", len(labels), path)
        return path
    if fmt != "folders":
        raise ValueError(f"Unknown export format {fmt!r}; use 'csv' or 'folders'.")

    written = 0
    for _, row in labels.iterrows():
        target_dir = out / row["label"]
        target_dir.mkdir(parents=True, exist_ok=True)
        source = config.resolve(row["path"])
        target = target_dir / f"{row['sha1'][:8]}_{source.name}"
        if target.exists():
            continue
        try:
            target.symlink_to(source.resolve())
        except OSError:
            import shutil

            shutil.copy2(source, target)
        written += 1
    log.info("Exported %d labels to folders under %s", written, out)
    return out
