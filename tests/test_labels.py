"""Label event log, undo and effective training labels (M4 acceptance)."""

from __future__ import annotations

from pathlib import Path

from _imagegen import drawing_image, photo_image
from pvd.config import Config
from pvd.imageio import sha1_file
from pvd.labels import LabelEvent, LabelStore, effective_labels, run_export
from pvd.scan import load_manifest, run_scan


def test_add_batch_and_current(tmp_path: Path) -> None:
    store = LabelStore(tmp_path / "events.jsonl")
    batch = store.add_batch(
        [LabelEvent(sha1="a", label="drawing"), LabelEvent(sha1="b", label="photo")]
    )
    assert batch == "b0001"
    assert store.current() == {"a": "drawing", "b": "photo"}


def test_undo_reverts_last_batch(tmp_path: Path) -> None:
    store = LabelStore(tmp_path / "events.jsonl")
    store.add_batch([LabelEvent(sha1="a", label="drawing")])
    store.add_batch([LabelEvent(sha1="c", label="skip")])

    assert store.undo_last() == "b0002"
    assert "c" not in store.current()
    assert store.current() == {"a": "drawing"}

    assert store.undo_last() == "b0001"
    assert store.current() == {}
    assert store.undo_last() is None


def test_persistence_across_reload(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    first = LabelStore(path)
    first.add_batch([LabelEvent(sha1="a", label="drawing")])
    second = LabelStore(path)
    assert second.current() == {"a": "drawing"}


def test_later_event_overrides_earlier(tmp_path: Path) -> None:
    store = LabelStore(tmp_path / "events.jsonl")
    store.add_batch([LabelEvent(sha1="a", label="drawing")])
    store.add_batch([LabelEvent(sha1="a", label="photo", source="audit_fix")])
    assert store.current() == {"a": "photo"}


def _seed_project(cfg: Config) -> None:
    (cfg.seed_dir / "photo").mkdir(parents=True, exist_ok=True)
    (cfg.seed_dir / "drawing").mkdir(parents=True, exist_ok=True)
    photo_image(seed=1).save(cfg.seed_dir / "photo" / "p.png")
    drawing_image(seed=1).save(cfg.seed_dir / "drawing" / "d.png")
    (cfg.test_dir / "photo").mkdir(parents=True, exist_ok=True)
    (cfg.test_dir / "drawing").mkdir(parents=True, exist_ok=True)
    photo_image(seed=2).save(cfg.test_dir / "photo" / "p.png")
    drawing_image(seed=2).save(cfg.test_dir / "drawing" / "d.png")


def test_effective_labels_combines_seed_and_overrides(tmp_project: Config) -> None:
    _seed_project(tmp_project)
    pool_image = tmp_project.pool_dir / "x.png"
    photo_image(seed=9).save(pool_image)
    run_scan(tmp_project)
    manifest = load_manifest(tmp_project)

    labels = effective_labels(tmp_project, manifest)
    assert set(labels["label"]) == {"photo", "drawing"}
    # Test images never appear in the training labels.
    test_shas = set(manifest.loc[manifest["split"] == "test", "sha1"])
    assert not (set(labels["sha1"]) & test_shas)

    store = LabelStore(tmp_project.labels_file)
    pool_sha = sha1_file(pool_image)
    store.add_batch([LabelEvent(sha1=pool_sha, label="drawing")])
    labels = effective_labels(tmp_project, manifest)
    assert pool_sha in set(labels["sha1"])


def test_skip_and_bad_labels_excluded(tmp_project: Config) -> None:
    _seed_project(tmp_project)
    pool_dir = tmp_project.pool_dir
    photo_image(seed=11).save(pool_dir / "junk.png")
    run_scan(tmp_project)
    manifest = load_manifest(tmp_project)
    store = LabelStore(tmp_project.labels_file)
    store.add_batch([LabelEvent(sha1=sha1_file(pool_dir / "junk.png"), label="skip")])
    labels = effective_labels(tmp_project, manifest)
    assert sha1_file(pool_dir / "junk.png") not in set(labels["sha1"])


def test_export_csv_and_folders(tmp_project: Config) -> None:
    _seed_project(tmp_project)
    run_scan(tmp_project)
    csv_path = run_export(tmp_project, fmt="csv", out=tmp_project.project_root / "export")
    assert csv_path.exists()
    assert "sha1" in csv_path.read_text(encoding="utf-8")

    folders = run_export(tmp_project, fmt="folders", out=tmp_project.project_root / "export2")
    assert (folders / "photo").exists()
    assert (folders / "drawing").exists()
