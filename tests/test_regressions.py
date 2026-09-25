"""Regression tests for the bugs found in the full code review.

Each test names the finding it guards. They use only the ``fake`` /
``fake_text`` backbones and synthetic images, so nothing is downloaded.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from _imagegen import drawing_image, photo_image, zero_byte
from pvd.config import Config, load_config
from pvd.features import (
    embedding_cache_path,
    feature_key,
    get_features,
    load_embedding_cache,
    run_embed,
)
from pvd.labels import LabelEvent, LabelStore, effective_labels
from pvd.scan import load_manifest, run_scan


def _fake(cfg: Config, backbone: str = "fake") -> Config:
    cfg.backbone.id = backbone
    cfg.num_workers = 0
    return cfg


def _seed(cfg: Config, n: int = 6, test: int = 3, pool: int = 0, start: int = 0) -> None:
    """Distinct seeds per split so no image is byte-identical across splits."""
    for cls, make in (("photo", photo_image), ("drawing", drawing_image)):
        (cfg.seed_dir / cls).mkdir(parents=True, exist_ok=True)
        (cfg.test_dir / cls).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            make(seed=start + i).save(cfg.seed_dir / cls / f"{cls}{i}.png")
        for i in range(test):
            make(seed=start + 500 + i).save(cfg.test_dir / cls / f"{cls}{i}.png")
    cfg.pool_dir.mkdir(parents=True, exist_ok=True)
    for i in range(pool):
        make = photo_image if i % 2 else drawing_image
        make(seed=start + 900 + i).save(cfg.pool_dir / f"pool{i}.png")


# --------------------------------------------------------------------------- #
# #1 project_root: the shipped relative "." root
# --------------------------------------------------------------------------- #
def test_scan_works_with_relative_project_root(tmp_path: Path, monkeypatch) -> None:
    project = tmp_path / "proj"
    (project / "data" / "pool").mkdir(parents=True)
    photo_image(seed=1).save(project / "data" / "pool" / "a.png")
    (project / "config.yaml").write_text(yaml.safe_dump({"project_root": "."}))

    # Run from a *different* directory: "." must mean the config file's folder.
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    cfg = load_config(project / "config.yaml")
    assert cfg.project_root == project.absolute()

    run_scan(cfg)
    manifest = load_manifest(cfg)
    assert manifest["path"].tolist() == ["data/pool/a.png"]


def test_default_config_file_stays_portable(tmp_path: Path) -> None:
    from pvd.config import write_default_config

    write_default_config(tmp_path / "config.yaml")
    assert yaml.safe_load((tmp_path / "config.yaml").read_text())["project_root"] == "."


def test_config_default_root_is_absolute() -> None:
    assert Config().project_root.is_absolute()


# --------------------------------------------------------------------------- #
# #3 torn last line in the label log
# --------------------------------------------------------------------------- #
def test_append_after_torn_line_keeps_new_event(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    store = LabelStore(path)
    store.add_batch([LabelEvent(sha1="a", label="photo")])
    with open(path, "a", encoding="utf-8") as handle:
        handle.write('{"sha1": "x", "la')  # crash mid-write: no newline

    store = LabelStore(path)
    store.add_batch([LabelEvent(sha1="b", label="drawing")])
    assert LabelStore(path).current() == {"a": "photo", "b": "drawing"}


def test_timestamps_use_z_suffix(tmp_path: Path) -> None:
    store = LabelStore(tmp_path / "events.jsonl")
    store.add_batch([LabelEvent(sha1="a", label="photo")])
    line = json.loads((tmp_path / "events.jsonl").read_text().splitlines()[0])
    assert line["ts"].endswith("Z")


def test_undo_depth_limits_consecutive_undos(tmp_path: Path) -> None:
    store = LabelStore(tmp_path / "events.jsonl")
    for sha in "abc":
        store.add_batch([LabelEvent(sha1=sha, label="photo")])
    assert store.undo_last(max_depth=2) is not None
    assert store.undo_last(max_depth=2) is not None
    assert store.undo_last(max_depth=2) is None
    assert store.current() == {"a": "photo"}


# --------------------------------------------------------------------------- #
# #14 test images never train; audit_fix overrides seed; renamed classes
# --------------------------------------------------------------------------- #
def test_logged_label_on_test_image_is_ignored(tmp_project: Config, caplog) -> None:
    cfg = _fake(tmp_project)
    _seed(cfg)
    run_scan(cfg)
    manifest = load_manifest(cfg)
    test_sha = manifest.loc[manifest["split"] == "test", "sha1"].iloc[0]
    LabelStore(cfg.labels_file).add_batch([LabelEvent(sha1=test_sha, label="photo")])

    with caplog.at_level(logging.WARNING):
        labels = effective_labels(cfg, manifest)
    assert test_sha not in set(labels["sha1"])
    assert "test split" in caplog.text


def test_audit_fix_overrides_seed_folder_label(tmp_project: Config) -> None:
    cfg = _fake(tmp_project)
    _seed(cfg)
    run_scan(cfg)
    manifest = load_manifest(cfg)
    seed_photo = manifest[(manifest["split"] == "seed") & (manifest["folder_label"] == "photo")].iloc[0]
    LabelStore(cfg.labels_file).add_batch(
        [LabelEvent(sha1=seed_photo["sha1"], label="drawing", source="audit_fix")]
    )
    labels = effective_labels(cfg, manifest).set_index("sha1")["label"]
    assert labels[seed_photo["sha1"]] == "drawing"


# --------------------------------------------------------------------------- #
# #2, #4, #13 review session: audit mode, quick retrain, minimum labels
# --------------------------------------------------------------------------- #
def _review_project(cfg: Config) -> None:
    _fake(cfg)
    cfg.review.batch_size = 6
    cfg.review.random_fraction = 0.5
    cfg.review.diversity = False
    _seed(cfg, n=8, test=3, pool=20)
    # Two deliberately wrong seed labels, so audit has suspects.
    drawing_image(seed=777).save(cfg.seed_dir / "photo" / "wrong_d.png")
    photo_image(seed=778).save(cfg.seed_dir / "drawing" / "wrong_p.png")
    run_scan(cfg)


def test_confirming_untouched_audit_batch_changes_nothing(tmp_project: Config) -> None:
    from pvd.review_app import ReviewSession

    _review_project(tmp_project)
    session = ReviewSession(tmp_project)
    before = effective_labels(tmp_project, session.manifest).set_index("sha1")["label"].to_dict()

    batch = session.set_mode("audit")
    assert batch, "expected label-error suspects"
    for position, item in enumerate(batch):
        # Untouched audit items show and default to their *stored* label.
        assert session.current_label(position) == item.stored_label
    summary = session.confirm()

    assert summary["confirmed"] == 0
    after = effective_labels(tmp_project, session.manifest).set_index("sha1")["label"].to_dict()
    assert after == before
    assert session.mode == "active"


def test_audit_correction_writes_audit_fix(tmp_project: Config) -> None:
    from pvd.review_app import ReviewSession

    _review_project(tmp_project)
    session = ReviewSession(tmp_project)
    batch = session.set_mode("audit")
    item = batch[0]
    new_label = next(c for c in tmp_project.classes if c != item.stored_label)
    session.set_label(0, new_label)
    session.confirm()

    events = [e for e in LabelStore(tmp_project.labels_file).history() if e.sha1 == item.sha1]
    assert events[-1].source == "audit_fix"
    assert events[-1].label == new_label
    assert events[-1].prev == item.stored_label


def test_confirm_uses_quick_retrain_between_cv_runs(tmp_project: Config, monkeypatch) -> None:
    import pvd.review_app as review_app

    _review_project(tmp_project)
    tmp_project.review.cv_every = 3
    calls = {"cv": 0, "quick": 0}
    real_train, real_quick = review_app.train_probe, review_app.make_estimator

    def counting_train(*args, **kwargs):
        calls["cv"] += 1
        return real_train(*args, **kwargs)

    def counting_quick(*args, **kwargs):
        calls["quick"] += 1
        return real_quick(*args, **kwargs)

    monkeypatch.setattr(review_app, "train_probe", counting_train)
    # train_probe builds its estimators inside pvd.probe, so this counts only quick refits.
    monkeypatch.setattr(review_app, "make_estimator", counting_quick)

    session = review_app.ReviewSession(tmp_project)
    assert calls == {"cv": 1, "quick": 0}  # startup: one CV search
    session.confirm()
    session.confirm()
    assert calls == {"cv": 1, "quick": 2}
    session.confirm()  # third batch reaches cv_every
    assert calls["cv"] == 2


def test_probe_needs_minimum_labels_per_class(tmp_project: Config) -> None:
    from pvd.review_app import ReviewSession

    cfg = _fake(tmp_project)
    cfg.review.batch_size = 4
    _seed(cfg, n=3, test=2, pool=6)  # 3 < MIN_TRAIN_PER_CLASS
    run_scan(cfg)
    session = ReviewSession(cfg)
    assert session.model_kind == "none"  # fake backbone has no text encoder
    assert "needs 5 labels per class" in session.model_reason
    assert "no text encoder" in session.model_reason


def test_zeroshot_fallback_with_text_backbone_and_tiles(tmp_project: Config) -> None:
    from pvd.review_app import ReviewSession

    cfg = _fake(tmp_project, "fake_text")
    cfg.backbone.tiles.enabled = True
    cfg.backbone.tiles.count = 2
    cfg.review.batch_size = 4
    _seed(cfg, n=2, test=2, pool=6)
    run_scan(cfg)
    run_embed(cfg)
    session = ReviewSession(cfg)  # used to crash: full (global+tiles) matrix vs text centroids
    assert session.model_kind == "zeroshot"
    assert session.probs.shape == (len(session.ok), 2)
    assert np.allclose(session.probs.sum(axis=1), 1.0)


def test_quick_mode_cycle_starts_from_model_guess(tmp_project: Config) -> None:
    from pvd.review_app import ReviewSession

    _review_project(tmp_project)
    session = ReviewSession(tmp_project)
    pred = session.batch[0].model_pred
    other = next(c for c in tmp_project.classes if c != pred)
    assert session.label_cycle(0) == [pred, other, "skip"]
    assert session.cycle_label(0) == other
    assert session.cycle_label(0) == "skip"
    assert session.cycle_label(0) == pred


def test_random_mode_marks_items_random(tmp_project: Config) -> None:
    from pvd.review_app import ReviewSession

    _review_project(tmp_project)
    session = ReviewSession(tmp_project)
    batch = session.set_mode("random")
    assert batch and all(item.sampled_as == "random" for item in batch)


def test_undo_reports_and_clears_last_batch(tmp_project: Config) -> None:
    from pvd.review_app import ReviewSession

    _review_project(tmp_project)
    session = ReviewSession(tmp_project)
    session.confirm()
    assert session.last_batch_summary is not None
    undone = session.undo()
    assert undone is not None
    assert session.last_batch_summary is None
    assert undone in session.status_markdown()


def test_empty_pool_message(tmp_project: Config) -> None:
    from pvd.review_app import ReviewSession

    cfg = _fake(tmp_project)
    _seed(cfg, n=6, test=2, pool=0)
    run_scan(cfg)
    session = ReviewSession(cfg)
    assert "No pool images found" in session.status_markdown()


def test_three_class_review_and_undo_restores_relabel(tmp_project: Config) -> None:
    from pvd.review_app import ReviewSession

    cfg = _fake(tmp_project)
    cfg.classes = ["photo", "drawing", "sketch"]
    cfg.review.batch_size = 3
    cfg.review.diversity = False
    _seed(cfg, n=6, test=2, pool=12)
    (cfg.seed_dir / "sketch").mkdir(parents=True)
    for i in range(6):
        drawing_image(seed=300 + i).save(cfg.seed_dir / "sketch" / f"s{i}.png")
    run_scan(cfg)

    session = ReviewSession(cfg)
    first = session.batch[0].sha1
    session.set_label(0, "sketch")
    session.confirm()
    assert session.store.current()[first] == "sketch"

    # Relabel the same image in a later audit-style correction, then undo it.
    LabelStore(cfg.labels_file)  # log readable by a fresh store
    session.store.add_batch([LabelEvent(sha1=first, label="photo", source="audit_fix")])
    assert session.store.current()[first] == "photo"
    session.undo()
    assert session.store.current()[first] == "sketch"


# --------------------------------------------------------------------------- #
# #5, #6, #7, #9 feature recipe and failures
# --------------------------------------------------------------------------- #
def test_feature_key_includes_every_recipe_setting(tmp_project: Config) -> None:
    cfg = _fake(tmp_project)
    cfg.backbone.tiles.enabled = True
    assert feature_key("clip_b32", cfg) == "clip_b32+tiles5"
    cfg.backbone.tiles.max_side = 1024
    assert feature_key("clip_b32", cfg) == "clip_b32+tiles5x1024"
    cfg.backbone.tiles.enabled = False
    cfg.image.alpha_background = (0, 0, 0)
    assert feature_key("clip_b32", cfg) == "clip_b32+bg000000"


def test_undecodable_image_is_masked_and_marked_broken(tmp_project: Config) -> None:
    cfg = _fake(tmp_project)
    _seed(cfg, n=2, test=0)
    run_scan(cfg)
    manifest = load_manifest(cfg)
    victim = manifest.iloc[0]
    # Corrupt the file *after* scan (scan saw it as ok).
    cfg.resolve(victim["path"]).write_bytes(b"not an image at all")

    shas = manifest["sha1"].tolist()
    paths = [str(cfg.resolve(p)) for p in manifest["path"]]
    X, valid = get_features(cfg, shas, paths, return_mask=True)
    assert not valid[0] and valid[1:].all()
    assert np.allclose(X[0], 0)

    run_embed(cfg)
    after = load_manifest(cfg).set_index("sha1")
    assert after.loc[victim["sha1"], "status"] == "broken"


def test_missing_file_is_not_remembered_as_failure(tmp_project: Config) -> None:
    cfg = _fake(tmp_project)
    _seed(cfg, n=1, test=0)
    run_scan(cfg)
    manifest = load_manifest(cfg)
    path = cfg.resolve(manifest.iloc[0]["path"])
    data = path.read_bytes()
    path.unlink()
    _, valid = get_features(cfg, manifest["sha1"].tolist(), [str(cfg.resolve(p)) for p in manifest["path"]], return_mask=True)
    assert not valid[0]
    path.write_bytes(data)
    _, valid = get_features(cfg, manifest["sha1"].tolist(), [str(cfg.resolve(p)) for p in manifest["path"]], return_mask=True)
    assert valid.all()


def test_deepghs_block_added_without_tiles(tmp_project: Config, monkeypatch) -> None:
    import sys
    import types

    validate = types.ModuleType("imgutils.validate")
    validate.anime_real_score = lambda path, **kw: {"anime": 0.25, "real": 0.75}
    monkeypatch.setitem(sys.modules, "imgutils", types.ModuleType("imgutils"))
    monkeypatch.setitem(sys.modules, "imgutils.validate", validate)

    cfg = _fake(tmp_project)
    cfg.backbone.extra_features.deepghs_real = True
    _seed(cfg, n=1, test=0)
    run_scan(cfg)
    run_embed(cfg)
    mapping, meta = load_embedding_cache(cfg, feature_key("fake", cfg))
    vector = next(iter(mapping.values()))
    assert meta["block_sizes"] == {"global": 128, "tiles": 0, "extra": 1}
    assert vector.shape == (129,)
    assert np.isclose(vector[-1], 2 * 0.75 - 1)


def test_deepghs_unknown_keys_raise(monkeypatch) -> None:
    import sys
    import types

    validate = types.ModuleType("imgutils.validate")
    validate.anime_real_score = lambda path, **kw: {"a": 0.1, "b": 0.9}
    monkeypatch.setitem(sys.modules, "imgutils", types.ModuleType("imgutils"))
    monkeypatch.setitem(sys.modules, "imgutils.validate", validate)
    from pvd.backbones.deepghs import real_score

    with pytest.raises(RuntimeError, match="Unexpected deepghs result keys"):
        real_score("x.png")


def test_deepghs_missing_gives_clean_cli_error(tmp_project: Config, monkeypatch) -> None:
    import sys

    from typer.testing import CliRunner

    from pvd.cli import app

    monkeypatch.setitem(sys.modules, "imgutils", None)
    (tmp_project.project_root / "config.yaml").write_text(
        yaml.safe_dump({"project_root": str(tmp_project.project_root)})
    )
    result = CliRunner().invoke(
        app, ["--config", str(tmp_project.project_root / "config.yaml"), "baseline-deepghs"]
    )
    assert result.exit_code == 1
    assert "pip install" in result.output
    assert "Traceback" not in result.output


def test_checkpoint_shards_resume_and_merge(tmp_project: Config, monkeypatch) -> None:
    import pvd.features as features
    from pvd.backbones.fake_backbone import FakeBackbone

    cfg = _fake(tmp_project)
    cfg.backbone.batch_size = 1
    monkeypatch.setattr(features, "CHECKPOINT_EVERY_BATCHES", 2)
    _seed(cfg, n=3, test=0)
    run_scan(cfg)
    manifest = load_manifest(cfg)
    shas = manifest["sha1"].tolist()
    paths = [str(cfg.resolve(p)) for p in manifest["path"]]
    key = feature_key("fake", cfg)

    original = FakeBackbone.encode
    state = {"calls": 0, "crash_after": 4}

    def counting_encode(self, batch):
        state["calls"] += 1
        if state["calls"] > state["crash_after"]:
            raise KeyboardInterrupt  # simulate the process dying mid-run
        return original(self, batch)

    monkeypatch.setattr(FakeBackbone, "encode", counting_encode)
    with pytest.raises(KeyboardInterrupt):
        get_features(cfg, shas, paths)
    assert not embedding_cache_path(cfg, key).exists()
    partial, _ = load_embedding_cache(cfg, key)
    assert len(partial) == 4  # two checkpoint shards of two images each

    # Resume: only the remaining two images are encoded, then shards merge.
    state.update(calls=0, crash_after=10**9)
    X = get_features(cfg, shas, paths)
    assert state["calls"] == 2
    assert embedding_cache_path(cfg, key).exists()
    assert not list(cfg.embeddings_dir.glob(f"{key}.part-*.npz"))
    assert not np.allclose(X, 0)


# --------------------------------------------------------------------------- #
# #8 similarity uses the unit-norm global block
# --------------------------------------------------------------------------- #
def test_sampler_tops_up_when_diversity_rejects() -> None:
    from pvd.sampler import next_batch

    cfg = Config(review={"batch_size": 6, "random_fraction": 0.0, "diversity": True})
    features = np.ones((10, 4), dtype=np.float32) * 2.0  # all identical, not unit-norm
    probs = np.column_stack([np.linspace(0.5, 1, 10), 1 - np.linspace(0.5, 1, 10)])
    batch = next_batch(probs, list(range(10)), features, cfg, np.random.default_rng(0))
    kinds = [kind for _, kind in batch]
    assert len(batch) == 6
    assert kinds.count("uncertain") == 1  # the rest were near-duplicates


# --------------------------------------------------------------------------- #
# #10 eval with reordered classes; #9 bundle recipe
# --------------------------------------------------------------------------- #
def test_eval_is_correct_after_classes_are_reordered(tmp_project: Config) -> None:
    from pvd.evaluate import evaluate_bundle
    from pvd.models import load_any as load
    from pvd.probe import run_train

    cfg = _fake(tmp_project)
    _seed(cfg, n=8, test=4)
    run_scan(cfg)
    bundle = load(run_train(cfg))
    before = evaluate_bundle(cfg, bundle).balanced_accuracy

    cfg.classes = ["drawing", "photo"]
    cfg.zeroshot.prompts = {k: cfg.zeroshot.prompts[k] for k in cfg.classes}
    after = evaluate_bundle(cfg, bundle).balanced_accuracy
    assert before > 0.9
    assert after == pytest.approx(before)


def test_predict_uses_bundle_recipe_not_current_config(tmp_project: Config) -> None:
    from pvd.predict import run_predict
    from pvd.models import load_any as load
    from pvd.probe import run_train

    cfg = _fake(tmp_project)
    _seed(cfg, n=8, test=2)
    run_scan(cfg)
    bundle = load(run_train(cfg))
    assert bundle.feature_key == "fake"
    assert bundle.feature_recipe["tiles"]["enabled"] is False

    # Turning tiles on afterwards must not change what the model is fed.
    cfg.backbone.tiles.enabled = True
    src = cfg.project_root / "inbox"
    src.mkdir()
    photo_image(seed=4000).save(src / "p.png")
    frame = pd.read_csv(run_predict(cfg, src, cfg.project_root / "out", mode="csv_only"))
    assert frame["pred"].tolist() == ["photo"]


def test_train_probe_requires_every_class() -> None:
    from pvd.probe import train_probe

    cfg = Config(classes=["photo", "drawing", "sketch"])
    X = np.random.default_rng(0).normal(size=(10, 4))
    y = np.array([0] * 5 + [1] * 5)
    with pytest.raises(ValueError, match="sketch"):
        train_probe(X, y, cfg, classes=cfg.classes, backbone_id="fake")


# --------------------------------------------------------------------------- #
# #12 classes/prompts
# --------------------------------------------------------------------------- #
def test_renamed_classes_validate_without_prompts() -> None:
    cfg = Config(classes=["real", "anime"])
    assert cfg.zeroshot.prompts == {}


def test_extra_class_keeps_default_prompts_for_known_ones() -> None:
    cfg = Config(classes=["photo", "drawing", "ai_generated"])
    assert set(cfg.zeroshot.prompts) == {"photo", "drawing"}


def test_reserved_class_names_rejected() -> None:
    with pytest.raises(ValueError, match="reserved"):
        Config(classes=["photo", "unsure"])


# --------------------------------------------------------------------------- #
# #11 fine-tune head split
# --------------------------------------------------------------------------- #
def test_finetune_head_is_only_the_classifier() -> None:
    timm = pytest.importorskip("timm")
    from pvd.finetune import _split_parameters

    model = timm.create_model("convnext_tiny", pretrained=False, num_classes=2)
    head, backbone = _split_parameters(model)
    assert len(head) == 2  # head.fc.weight + head.fc.bias, not every mlp.fc1/fc2
    assert sum(p.numel() for p in head) == 768 * 2 + 2


# --------------------------------------------------------------------------- #
# #18 scan edge cases
# --------------------------------------------------------------------------- #
def test_stray_file_in_test_root_is_flagged_not_fatal(tmp_project: Config) -> None:
    from pvd.evaluate import resolve_test_set

    cfg = _fake(tmp_project)
    _seed(cfg, n=2, test=2)
    photo_image(seed=999).save(cfg.test_dir / "stray.png")
    run_scan(cfg)
    manifest = load_manifest(cfg).set_index("path")
    assert manifest.loc["data/test/stray.png", "status"] == "bad_label"
    assert len(resolve_test_set(cfg).shas) == 4


def test_typo_folder_does_not_hide_pool_copy(tmp_project: Config) -> None:
    cfg = _fake(tmp_project)
    (cfg.seed_dir / "phtoo").mkdir(parents=True)
    photo_image(seed=5).save(cfg.seed_dir / "phtoo" / "x.png")
    photo_image(seed=5).save(cfg.pool_dir / "x.png")
    run_scan(cfg)
    manifest = load_manifest(cfg).set_index("path")
    assert manifest.loc["data/seed/phtoo/x.png", "status"] == "bad_label"
    assert manifest.loc["data/pool/x.png", "status"] == "ok"


def test_rescan_reuses_unchanged_files(tmp_project: Config, monkeypatch) -> None:
    import pvd.scan as scan_module

    cfg = _fake(tmp_project)
    _seed(cfg, n=2, test=1)
    run_scan(cfg)
    first = load_manifest(cfg)

    hashed: list[str] = []
    real_sha1 = scan_module.sha1_file
    monkeypatch.setattr(scan_module, "sha1_file", lambda p: hashed.append(str(p)) or real_sha1(p))
    photo_image(seed=12345).save(cfg.pool_dir / "new.png")
    run_scan(cfg)
    assert [Path(p).name for p in hashed] == ["new.png"]
    second = load_manifest(cfg)
    assert second[second["path"] != "data/pool/new.png"]["sha1"].tolist() == first["sha1"].tolist()

    hashed.clear()
    run_scan(cfg, full=True)
    assert len(hashed) == len(second)


def test_manifest_integer_columns_stay_integers(tmp_project: Config) -> None:
    cfg = _fake(tmp_project)
    _seed(cfg, n=1, test=0)
    zero_byte(cfg.pool_dir / "broken.jpg")
    run_scan(cfg)
    manifest = load_manifest(cfg)
    assert str(manifest["width"].dtype) == "Int64"
    assert str(manifest["has_alpha"].dtype) == "boolean"


# --------------------------------------------------------------------------- #
# #19 predict edge cases
# --------------------------------------------------------------------------- #
def _trained(cfg: Config) -> None:
    _fake(cfg)
    _seed(cfg, n=8, test=2)
    run_scan(cfg)
    from pvd.probe import run_train

    run_train(cfg)


def test_predict_empty_source(tmp_project: Config) -> None:
    from pvd.predict import run_predict

    _trained(tmp_project)
    src = tmp_project.project_root / "empty"
    src.mkdir()
    frame = pd.read_csv(run_predict(tmp_project, src, tmp_project.project_root / "out"))
    assert frame.empty


def test_predict_same_name_in_subfolders_and_relative_paths(tmp_project: Config) -> None:
    from pvd.predict import run_predict

    _trained(tmp_project)
    src = tmp_project.project_root / "inbox"
    (src / "a").mkdir(parents=True)
    (src / "b").mkdir(parents=True)
    photo_image(seed=7001).save(src / "a" / "x.png")
    photo_image(seed=7002).save(src / "b" / "x.png")
    dst = tmp_project.project_root / "out"
    frame = pd.read_csv(run_predict(tmp_project, src, dst, mode="copy", threshold=0.5))
    assert sorted(frame["path"]) == ["a/x.png", "b/x.png"]
    assert len(set(frame["dest"])) == 2
    assert all((dst / d).exists() for d in frame["dest"])

    # A second run never overwrites: file count doubles, first files unchanged.
    first = {d: (dst / d).read_bytes() for d in frame["dest"]}
    run_predict(tmp_project, src, dst, mode="copy", threshold=0.5)
    assert len(list(dst.rglob("*.png"))) == 4
    assert all((dst / d).read_bytes() == data for d, data in first.items())


@pytest.mark.skipif(os.name == "nt", reason="symlinks need privileges on Windows")
def test_predict_skips_symlinked_inputs(tmp_project: Config) -> None:
    from pvd.predict import run_predict

    _trained(tmp_project)
    src = tmp_project.project_root / "inbox"
    src.mkdir()
    secret = tmp_project.project_root / "secret.txt"
    secret.write_text("private")
    (src / "photo.jpg").symlink_to(secret)
    photo_image(seed=8001).save(src / "real.png")
    frame = pd.read_csv(run_predict(tmp_project, src, tmp_project.project_root / "out"))
    assert frame["path"].tolist() == ["real.png"]


def test_predict_does_not_overwrite_previous_csv(tmp_project: Config) -> None:
    from pvd.predict import run_predict

    _trained(tmp_project)
    src = tmp_project.project_root / "inbox"
    src.mkdir()
    photo_image(seed=9001).save(src / "p.png")
    dst = tmp_project.project_root / "out"
    first = run_predict(tmp_project, src, dst, mode="csv_only")
    second = run_predict(tmp_project, src, dst, mode="csv_only")
    assert first != second and first.exists() and second.exists()


# --------------------------------------------------------------------------- #
# #22, #24 security hardening
# --------------------------------------------------------------------------- #
def test_non_allowlisted_format_is_rejected(tmp_path: Path) -> None:
    from PIL import Image

    from pvd.imageio import ImageLoadError, load_rgb, probe_image

    # A real image in a format outside the allowlist, disguised as .jpg.
    disguised = tmp_path / "x.jpg"
    Image.new("RGB", (8, 8)).save(disguised, format="PCX")
    with pytest.raises(ImageLoadError):
        probe_image(disguised)
    with pytest.raises(ImageLoadError):
        load_rgb(disguised)


@pytest.mark.skipif(os.name == "nt", reason="symlinks need privileges on Windows")
def test_export_never_writes_through_dangling_symlink(tmp_project: Config) -> None:
    from pvd.labels import run_export

    cfg = _fake(tmp_project)
    _seed(cfg, n=1, test=0)
    run_scan(cfg)
    manifest = load_manifest(cfg)
    row = manifest[manifest["folder_label"] == "photo"].iloc[0]
    out = cfg.project_root / "export"
    (out / "photo").mkdir(parents=True)
    outside = cfg.project_root / "outside.png"
    name = f"{row['sha1'][:8]}_{Path(row['path']).name}"
    (out / "photo" / name).symlink_to(outside)  # dangling: target doesn't exist

    run_export(cfg, fmt="folders", out=out)
    assert not outside.exists()
