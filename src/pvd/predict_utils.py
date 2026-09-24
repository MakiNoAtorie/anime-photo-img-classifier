"""Small helpers for the predict command."""

from __future__ import annotations

from pathlib import Path


def unique_destination(target: Path, sha1: str, used: dict[str, Path] | None = None) -> Path:
    """Return a collision-free path; never overwrite an existing file (§9.13).

    On a name clash the first 8 characters of the content hash are appended, so
    distinct images with the same filename never overwrite each other.
    """
    used = used if used is not None else {}
    tag = sha1[:8] if isinstance(sha1, str) and sha1 else "unknown"
    if not target.exists() and target not in used.values():
        return target
    stem, suffix = target.stem, target.suffix
    candidate = target.with_name(f"{stem}_{tag}{suffix}")
    counter = 2
    while candidate.exists() or candidate in used.values():
        candidate = target.with_name(f"{stem}_{tag}_{counter}{suffix}")
        counter += 1
    return candidate
