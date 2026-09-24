"""Review-session state machine and Gradio app wiring (M5 acceptance)."""

from __future__ import annotations

import pytest

from _imagegen import drawing_image, photo_image
from pvd.config import Config
from pvd.review_app import ReviewSession, build_app
from pvd.scan import run_scan


def _project(cfg: Config) -> None:
    cfg.backbone.id = "fake"
    cfg.num_workers = 0
    cfg.review.batch_size = 6
    cfg.review.random_fraction = 0.5
    cfg.review.diversity = False
    for cls, make in (("photo", photo_image), ("drawing", drawing_image)):
        (cfg.seed_dir / cls).mkdir(parents=True, exist_ok=True)
        (cfg.test_dir / cls).mkdir(parents=True, exist_ok=True)
        for i in range(6):
            make(seed=i).save(cfg.seed_dir / cls / f"{cls}{i}.png")
        for i in range(3):
            make(seed=50 + i).save(cfg.test_dir / cls / f"{cls}{i}.png")
    for i in range(20):
        photo_image(seed=200 + i).save(cfg.pool_dir / f"pool{i}.png")


def test_batch_and_select(tmp_project: Config) -> None:
    _project(tmp_project)
    run_scan(tmp_project)
    session = ReviewSession(tmp_project)

    assert len(session.batch) == 6
    assert len(session.candidate_positions()) == 20
    item = session.select(0)
    assert item is not None
    # Selecting does not change the label.
    assert session.current_label(0) == item.model_pred
    assert session.changed(0) is False
    session.set_label(0, "drawing")
    assert session.changed(0) is True
    assert session.current_label(0) == "drawing"


def test_confirm_persists_and_advances(tmp_project: Config) -> None:
    _project(tmp_project)
    run_scan(tmp_project)
    session = ReviewSession(tmp_project)
    session.set_label(0, "drawing")
    summary = session.confirm()

    assert summary["confirmed"] == 6
    assert summary["changed"] == 1
    # All confirmed images are now labeled and out of the candidate pool.
    assert len(session.candidate_positions()) == 14

    # A fresh session (as if the app was restarted) sees the labels.
    reloaded = ReviewSession(tmp_project)
    assert len(reloaded.candidate_positions()) == 14
    assert reloaded.store.current()


def test_undo_makes_images_candidates_again(tmp_project: Config) -> None:
    _project(tmp_project)
    run_scan(tmp_project)
    session = ReviewSession(tmp_project)
    session.confirm()
    assert len(session.candidate_positions()) == 14

    undone = session.undo()
    assert undone is not None
    assert len(session.candidate_positions()) == 20
    assert session.store.current() == {}


def test_skip_is_recorded_but_not_training(tmp_project: Config) -> None:
    _project(tmp_project)
    run_scan(tmp_project)
    session = ReviewSession(tmp_project)
    for position in range(len(session.batch)):
        session.set_label(position, "skip")
    session.confirm()

    from pvd.labels import effective_labels

    labels = effective_labels(tmp_project, session.manifest)
    assert "skip" not in set(labels["label"])
    assert len(session.candidate_positions()) == 14


def test_status_mentions_metrics(tmp_project: Config) -> None:
    _project(tmp_project)
    run_scan(tmp_project)
    session = ReviewSession(tmp_project)
    status = session.status_markdown()
    assert "Test balanced accuracy" in status
    assert "Model: linear probe" in status


def test_build_app_constructs(tmp_project: Config) -> None:
    pytest.importorskip("gradio")
    _project(tmp_project)
    run_scan(tmp_project)
    session = ReviewSession(tmp_project)
    demo = build_app(session)
    assert demo is not None
