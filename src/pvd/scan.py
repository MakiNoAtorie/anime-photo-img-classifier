"""Build the image manifest: discover, validate, hash and de-duplicate (G3, G6).

The manifest is a single table describing every image the project knows about.
Everything downstream (embedding, labels, splits) joins against it by ``sha1``.

Dedupe policy (from section 8.2): identical SHA-1 keeps the highest-priority copy
``test > seed > pool`` so the held-out set stays clean; the rest are marked
``duplicate``. If the same bytes appear under two different seed/test labels, both
are marked ``label_conflict`` and excluded from training.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import Config
from .imageio import ImageLoadError, probe_image, sha1_file
from .imageio import load_rgb

log = logging.getLogger(__name__)

# Statuses that are excluded from embeddings, training and review batches.
EXCLUDED_STATUSES = {"broken", "too_large", "duplicate", "label_conflict", "bad_label"}

SPLIT_PRIORITY = {"test": 0, "seed": 1, "pool": 2}

MANIFEST_COLUMNS = [
    "path",
    "sha1",
    "split",
    "folder_label",
    "tag",
    "source_dir",
    "width",
    "height",
    "format",
    "mode",
    "has_alpha",
    "n_frames",
    "status",
    "duplicate_of",
    "error",
]


def iter_images(root: Path, extensions: set[str]) -> list[Path]:
    """Return all files under ``root`` whose extension is in ``extensions``."""
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )


def _locate(rel_parts: tuple[str, ...], split: str) -> tuple[str | None, str | None, str]:
    """Return ``(folder_label, tag, source_dir)`` for a split-relative path."""
    if split == "pool":
        return None, None, rel_parts[0] if len(rel_parts) > 1 else "."
    folder_label = rel_parts[0] if len(rel_parts) > 1 else None
    nested = rel_parts[1:-1]
    if split == "test":
        tag = nested[0] if nested else None
        source_dir = nested[-1] if nested else "."
    else:
        tag = None
        source_dir = nested[-1] if nested else "."
    return folder_label, tag, source_dir


def _discover(config: Config) -> list[dict]:
    extensions = set(config.image.extensions)
    roots = {"pool": config.pool_dir, "seed": config.seed_dir, "test": config.test_dir}
    rows: list[dict] = []
    for split, root in roots.items():
        for path in iter_images(root, extensions):
            rel = path.relative_to(root).parts
            folder_label, tag, source_dir = _locate(rel, split)
            project_rel = path.relative_to(config.project_root).as_posix()
            row = {
                "path": project_rel,
                "sha1": sha1_file(path),
                "split": split,
                "folder_label": folder_label,
                "tag": tag,
                "source_dir": source_dir,
                "width": None,
                "height": None,
                "format": None,
                "mode": None,
                "has_alpha": None,
                "n_frames": None,
                "status": "ok",
                "duplicate_of": None,
                "error": None,
            }
            if folder_label is not None and folder_label not in config.classes:
                row["status"] = "bad_label"
                row["error"] = f"unknown label directory: {folder_label!r}"
                rows.append(row)
                continue
            try:
                info = probe_image(path)
                # Full decode to catch truncated/corrupt files that open fine.
                load_rgb(path, tuple(config.image.alpha_background), config.image.max_pixels)
                row.update(
                    width=info.width,
                    height=info.height,
                    format=info.format,
                    mode=info.mode,
                    has_alpha=info.has_alpha,
                    n_frames=info.n_frames,
                )
            except ImageLoadError as exc:
                row["status"] = exc.reason
                row["error"] = exc.message
            except Exception as exc:  # pragma: no cover - safety net
                row["status"] = "broken"
                row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
    return rows


def _dedupe(rows: list[dict]) -> None:
    """Apply priority dedupe and label-conflict detection in place."""
    by_sha: dict[str, list[dict]] = {}
    for row in rows:
        by_sha.setdefault(row["sha1"], []).append(row)

    for group in by_sha.values():
        if len(group) == 1:
            continue
        labels = {r["folder_label"] for r in group if r["folder_label"] is not None}
        if len(labels) > 1:
            for row in group:
                row["status"] = "label_conflict"
                row["error"] = f"same file labeled as {sorted(labels)}"
            continue
        # Prefer the highest-priority split, then files that are not broken.
        group.sort(
            key=lambda r: (
                SPLIT_PRIORITY.get(r["split"], 9),
                0 if r["status"] == "ok" else 1,
                r["path"],
            )
        )
        canonical = group[0]
        for row in group[1:]:
            row["status"] = "duplicate"
            row["duplicate_of"] = canonical["sha1"]
            row["error"] = None
            canonical.setdefault("duplicate_count", 0)


def scan(config: Config) -> pd.DataFrame:
    """Discover all images and return the manifest table (section 8.2)."""
    rows = _discover(config)
    _dedupe(rows)
    if not rows:
        return pd.DataFrame(columns=MANIFEST_COLUMNS)
    frame = pd.DataFrame(rows)
    for column in MANIFEST_COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    frame = frame[MANIFEST_COLUMNS].sort_values("path").reset_index(drop=True)
    return frame


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _format_share(frame: pd.DataFrame, column: str) -> str:
    lines = []
    for split in ("seed", "test"):
        sub = frame[frame["split"] == split]
        for cls, group in sub.groupby("folder_label"):
            total = len(group)
            if total == 0:
                continue
            counts = group[column].value_counts().head(4)
            share = ", ".join(f"{name}={count} ({count / total:.0%})" for name, count in counts.items())
            lines.append(f"  - {split}/{cls} ({total}): {share}")
    return "\n".join(lines) if lines else "  (no seed/test images)"


def build_report(config: Config, frame: pd.DataFrame) -> str:
    """Render a Markdown scan report with the data-integrity warnings."""
    ok = frame[frame["status"] == "ok"]
    lines: list[str] = []
    lines.append("# Scan report")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    lines.append("")
    lines.append(f"Total discovered: **{len(frame)}**  |  usable (`ok`): **{len(ok)}**")
    lines.append("")

    lines.append("## Counts by status")
    lines.append("")
    lines.append(frame["status"].value_counts().to_string())
    lines.append("")

    lines.append("## Counts by split / class")
    lines.append("")
    for (split, label), group in frame.groupby(["split", "folder_label"], dropna=False):
        name = label if isinstance(label, str) else "(none)"
        lines.append(f"- {split}/{name}: {len(group)}")
    lines.append("")

    lines.append("## Tags (test set)")
    lines.append("")
    tagged = frame[(frame["split"] == "test") & frame["tag"].notna()]
    if tagged.empty:
        lines.append("- (none)")
    else:
        for (label, tag), group in tagged.groupby(["folder_label", "tag"]):
            lines.append(f"- {label}/{tag}: {len(group)}")
    lines.append("")

    lines.append("## Early bias warnings")
    lines.append("")
    lines.append("File-format share (a model can learn 'PNG -> drawing' as a shortcut):")
    lines.append(_format_share(frame, "format"))
    lines.append("")
    lines.append("Transparent-alpha share:")
    lines.append(_format_share(frame.assign(alpha=frame["has_alpha"].map({True: "alpha", False: "opaque"})), "alpha"))
    lines.append("")

    problems = frame[frame["status"].isin(EXCLUDED_STATUSES)]
    if not problems.empty:
        lines.append("## Excluded files")
        lines.append("")
        for _, row in problems.iterrows():
            duplicate_of = row["duplicate_of"]
            error = row["error"]
            detail = f" -> duplicate of {str(duplicate_of)[:8]}" if isinstance(duplicate_of, str) else ""
            message = f" ({error})" if isinstance(error, str) and error else ""
            lines.append(f"- `{row['path']}` [{row['status']}]{detail}{message}")
        lines.append("")

    return "\n".join(lines)


def run_scan(config: Config) -> Path:
    """Scan, write ``manifest.parquet`` and a Markdown report."""
    frame = scan(config)
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    config.reports_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(config.manifest_path, index=False)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = config.reports_dir / f"{stamp}_scan.md"
    report_path.write_text(build_report(config, frame), encoding="utf-8")

    ok = int((frame["status"] == "ok").sum())
    log.info("Scanned %d files (%d usable) -> %s", len(frame), ok, config.manifest_path)
    log.info("Scan report: %s", report_path)
    return report_path


def load_manifest(config: Config) -> pd.DataFrame:
    """Load the manifest, with a clear error if ``scan`` has not been run."""
    if not config.manifest_path.exists():
        raise FileNotFoundError(
            f"No manifest at {config.manifest_path}. Run `pvd scan` first."
        )
    return pd.read_parquet(config.manifest_path)
