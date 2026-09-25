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
from .imageio import ImageLoadError, load_rgb, probe_image, sha1_file

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
    # Bookkeeping for incremental re-scans: an unchanged file (same size and
    # modification time) reuses its hash and decode result instead of being
    # read and decoded again.
    "size",
    "mtime_ns",
    "decode_status",
]

# Decode outcome is reused from the previous manifest only for these statuses.
_REUSABLE_DECODE = {"ok", "broken"}


def iter_images(root: Path, extensions: set[str], follow_symlinks: bool = True) -> list[Path]:
    """Return all files under ``root`` whose extension is in ``extensions``.

    ``follow_symlinks=False`` skips symlinked files; ``predict`` uses it for
    input folders it did not create, so a link to e.g. a private key is never
    copied into the output.
    """
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in extensions
        and (follow_symlinks or not path.is_symlink())
    )


def _locate(rel_parts: tuple[str, ...], split: str) -> tuple[str | None, str | None, str]:
    """Return ``(folder_label, tag, source_dir)`` for a split-relative path."""
    if split == "pool":
        return None, None, rel_parts[0] if len(rel_parts) > 1 else "."
    folder_label = rel_parts[0] if len(rel_parts) > 1 else None
    nested = rel_parts[1:-1]
    tag = nested[0] if split == "test" and nested else None
    return folder_label, tag, nested[-1] if nested else "."


def _previous_rows(config: Config) -> dict[str, dict]:
    """Previous manifest rows by path, for incremental re-scans."""
    if not config.manifest_path.exists():
        return {}
    try:
        previous = pd.read_parquet(config.manifest_path)
    except Exception as exc:  # noqa: BLE001 - a bad cache only costs a full scan
        log.warning("Ignoring unreadable previous manifest (%s); doing a full scan.", exc)
        return {}
    if "mtime_ns" not in previous.columns:
        return {}
    return {row["path"]: row for row in previous.to_dict("records")}


def _reuse(prev: dict | None, stat, config: Config) -> dict | None:
    """Cached hash/decode fields for an unchanged file, or None to recompute."""
    if not prev or prev.get("decode_status") not in _REUSABLE_DECODE:
        return None
    if prev.get("size") != stat.st_size or prev.get("mtime_ns") != stat.st_mtime_ns:
        return None
    fields = {k: prev.get(k) for k in ("sha1", "width", "height", "format", "mode", "has_alpha", "n_frames")}
    status, error = prev["decode_status"], (prev.get("error") if prev["decode_status"] != "ok" else None)
    width, height = fields["width"], fields["height"]
    if status == "ok" and pd.notna(width) and pd.notna(height) and int(width) * int(height) > config.image.max_pixels:
        status, error = "too_large", f"{int(width)}x{int(height)} exceeds max_pixels={config.image.max_pixels}"
    return {**fields, "status": status, "error": error}


def _discover(config: Config, incremental: bool = True) -> list[dict]:
    extensions = set(config.image.extensions)
    roots = {"pool": config.pool_dir, "seed": config.seed_dir, "test": config.test_dir}
    previous = _previous_rows(config) if incremental else {}
    rows: list[dict] = []
    reused = 0
    for split, root in roots.items():
        for path in iter_images(root, extensions):
            rel = path.relative_to(root).parts
            folder_label, tag, source_dir = _locate(rel, split)
            project_rel = config.relative(path)
            stat = path.stat()
            row = {
                "path": project_rel,
                "sha1": None,
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
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "decode_status": None,
            }
            cached = _reuse(previous.get(project_rel), stat, config)
            if cached is not None:
                row.update(cached)
                reused += 1
            else:
                row["sha1"] = sha1_file(path)
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
                    row["error"] = str(exc)
                except Exception as exc:  # pragma: no cover - safety net
                    row["status"] = "broken"
                    row["error"] = f"{type(exc).__name__}: {exc}"
            row["decode_status"] = row["status"]
            # Label checks are cheap and depend on the config, so never cached.
            if split != "pool" and folder_label is None:
                row["status"] = "bad_label"
                row["error"] = f"files in {split}/ must be inside a class folder, e.g. {split}/<class>/"
            elif folder_label is not None and folder_label not in config.classes:
                row["status"] = "bad_label"
                row["error"] = f"unknown label directory: {folder_label!r}"
            rows.append(row)
    if reused:
        log.info("Reused %d unchanged file(s) from the previous scan.", reused)
    return rows


def _dedupe(rows: list[dict]) -> None:
    """Apply priority dedupe and label-conflict detection in place.

    Rows with a bad label directory take no part: a typo folder must not hide
    (or conflict with) a valid copy of the same image elsewhere.
    """
    by_sha: dict[str, list[dict]] = {}
    for row in rows:
        if row["status"] == "bad_label":
            continue
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
        # Highest-priority split wins (test > seed > pool) so the test set stays clean.
        group.sort(key=lambda r: (SPLIT_PRIORITY.get(r["split"], 9), r["path"]))
        canonical = group[0]
        for row in group[1:]:
            row["status"] = "duplicate"
            row["duplicate_of"] = canonical["sha1"]
            row["error"] = None


def scan(config: Config, incremental: bool = True) -> pd.DataFrame:
    """Discover all images and return the manifest table (section 8.2)."""
    rows = _discover(config, incremental=incremental)
    _dedupe(rows)
    frame = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    frame = frame.sort_values("path").reset_index(drop=True)
    # Nullable dtypes keep ints as ints (not float64) when some rows are broken.
    for column in ("width", "height", "n_frames", "size", "mtime_ns"):
        frame[column] = frame[column].astype("Int64")
    frame["has_alpha"] = frame["has_alpha"].astype("boolean")
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


def run_scan(config: Config, full: bool = False) -> Path:
    """Scan, write ``manifest.parquet`` and a Markdown report.

    Unchanged files reuse the previous scan's results unless ``full`` is set.
    """
    frame = scan(config, incremental=not full)
    config.reports_dir.mkdir(parents=True, exist_ok=True)
    save_manifest(config, frame)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = config.reports_dir / f"{stamp}_scan.md"
    report_path.write_text(build_report(config, frame), encoding="utf-8")

    ok = int((frame["status"] == "ok").sum())
    log.info("Scanned %d files (%d usable) -> %s", len(frame), ok, config.manifest_path)
    log.info("Scan report: %s", report_path)
    return report_path


def save_manifest(config: Config, frame: pd.DataFrame) -> Path:
    """Write the manifest atomically (temp file + rename)."""
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = config.manifest_path.with_name(config.manifest_path.name + ".tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(config.manifest_path)
    return config.manifest_path


def load_manifest(config: Config) -> pd.DataFrame:
    """Load the manifest, with a clear error if ``scan`` has not been run."""
    if not config.manifest_path.exists():
        raise FileNotFoundError(
            f"No manifest at {config.manifest_path}. Run `pvd scan` first."
        )
    return pd.read_parquet(config.manifest_path)
