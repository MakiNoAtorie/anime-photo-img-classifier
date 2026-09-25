"""Backbone / feature-block ablation (M8, §9.10).

Trains the same probe recipe on different feature configurations and reports CV
and held-out test metrics side by side, plus a rough embedding-speed figure. The
recommendation is generated from the measured numbers, never hard-coded.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
import torch

from .config import Config
from .evaluate import evaluate_bundle, resolve_test_set, write_report
from .features import feature_recipe, get_features
from .labels import effective_labels
from .probe import train_probe
from .scan import load_manifest

log = logging.getLogger(__name__)

SPEED_SAMPLE = 16

# One report row; every metric is None until measured.
_EMPTY_ROW = {
    "cv_balanced_accuracy": None,
    "test_balanced_accuracy": None,
    "realistic_drawing": None,
    "filtered_photo": None,
    "embed_seconds_per_1k": None,
    "error": None,
}


@dataclass
class AblationSpec:
    """One row of the ablation grid: a backbone plus which optional feature blocks to add."""
    name: str
    backbone: str
    tiles: bool = False
    deepghs: bool = False


def default_specs(config: Config) -> list[AblationSpec]:
    """A sensible starting grid; add/remove entries freely."""
    backbones = ["clip_b32", "clip_l14", "siglip_so400m", "dinov2_s", "dinov2_b"]
    specs = [AblationSpec(name=b, backbone=b) for b in backbones]
    specs.append(AblationSpec(name="clip_b32+tiles5", backbone="clip_b32", tiles=True))
    specs.append(AblationSpec(name="clip_l14+tiles5", backbone="clip_l14", tiles=True))
    from .backbones.deepghs import deepghs_available

    if deepghs_available():
        specs.append(AblationSpec(name="clip_b32+deepghs", backbone="clip_b32", deepghs=True))
    return specs


def measure_embedding_speed(backbone, paths: list[str], config: Config, max_images: int = SPEED_SAMPLE) -> float | None:
    """Seconds to embed 1,000 images (encode only, warm model)."""
    from .imageio import load_rgb

    sample = paths[:max_images]
    if not sample:
        return None
    bg = tuple(config.image.alpha_background)
    tensors = [backbone.preprocess(load_rgb(p, bg, config.image.max_pixels)) for p in sample]
    started = time.perf_counter()
    for start in range(0, len(tensors), config.backbone.batch_size):
        batch = torch.stack(tensors[start : start + config.backbone.batch_size])
        backbone.encode(batch)
    elapsed = time.perf_counter() - started
    return elapsed / len(sample) * 1000.0


def _run_spec(config: Config, spec: AblationSpec, labels, test) -> dict:
    from .backbones import build_backbone
    from .device import pick_device

    cfg = config.model_copy(deep=True)
    cfg.backbone.id = spec.backbone
    cfg.backbone.tiles.enabled = spec.tiles
    cfg.backbone.extra_features.deepghs_real = spec.deepghs

    class_to_index = {name: i for i, name in enumerate(cfg.classes)}
    train_shas = labels["sha1"].tolist()
    train_paths = [str(cfg.resolve(p)) for p in labels["path"].tolist()]
    X, valid = get_features(cfg, train_shas, train_paths, backbone_id=spec.backbone, return_mask=True)
    X, labels = X[valid], labels[valid].reset_index(drop=True)
    train_paths = [p for p, ok in zip(train_paths, valid) if ok]
    y = np.array([class_to_index[label] for label in labels["label"]], dtype=int)
    bundle = train_probe(
        X,
        y,
        cfg,
        classes=list(cfg.classes),
        backbone_id=spec.backbone,
        recipe=feature_recipe(spec.backbone, cfg),
    )

    row = {**_EMPTY_ROW, "name": spec.name, "cv_balanced_accuracy": bundle.cv_balanced_accuracy}
    if test is not None:
        result = evaluate_bundle(cfg, bundle, test)
        row["test_balanced_accuracy"] = result.balanced_accuracy
        row["realistic_drawing"] = result.per_tag.get("realistic_drawing", {}).get("accuracy")
        row["filtered_photo"] = result.per_tag.get("filtered_photo", {}).get("accuracy")

    device = pick_device(cfg.device)
    backbone = build_backbone(spec.backbone, device)
    row["embed_seconds_per_1k"] = measure_embedding_speed(backbone, train_paths, cfg)
    return row


def _recommendation(rows: list[dict]) -> str:
    valid = [r for r in rows if r.get("error") is None]
    if not valid:
        return "No configuration completed successfully; nothing to recommend."
    key = "test_balanced_accuracy"
    if all(r[key] is None for r in valid):
        key = "cv_balanced_accuracy"
    best = max(valid, key=lambda r: (r[key] if r[key] is not None else -1))
    lines = [
        "## Recommendation",
        "",
        f"- Best `{key}` = **{best[key]:.3f}** with `{best['name']}`"
        + (f" ({best['embed_seconds_per_1k']:.2f}s/1k images)" if best["embed_seconds_per_1k"] else ""),
        "",
    ]
    if best["realistic_drawing"] is not None or best["filtered_photo"] is not None:
        lines.append(
            "- On the hard tags it scores "
            f"`realistic_drawing`={_fmt(best['realistic_drawing'])}, "
            f"`filtered_photo`={_fmt(best['filtered_photo'])}."
        )
        lines.append("")
    lines.append(
        "Pick the best accuracy you can afford; when two are close, prefer the "
        "cheaper/faster one. Tiles trade speed for texture detail."
    )
    return "\n".join(lines)


def _fmt(value) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def render_ablation(rows: list[dict]) -> str:
    """Markdown table of all configurations plus a recommendation computed from the numbers."""
    lines = ["# Ablation report", ""]
    lines.append("| configuration | CV balanced acc | test balanced acc | realistic_drawing | filtered_photo | s / 1k | note |")
    lines.append("|---|---|---|---|---|---|---|")
    for row in rows:
        note = row.get("error") or ""
        lines.append(
            "| {name} | {cv} | {test} | {rd} | {fp} | {speed} | {note} |".format(
                name=row["name"],
                cv=_fmt(row["cv_balanced_accuracy"]),
                test=_fmt(row["test_balanced_accuracy"]),
                rd=_fmt(row["realistic_drawing"]),
                fp=_fmt(row["filtered_photo"]),
                speed=_fmt(row["embed_seconds_per_1k"]),
                note=note[:60],
            )
        )
    lines.append("")
    lines.append(_recommendation(rows))
    return "\n".join(lines)


def run_ablation(config: Config, specs: list[AblationSpec] | None = None) -> Path:
    """Train one probe per feature configuration and write the ablation report."""
    manifest = load_manifest(config)
    labels = effective_labels(config, manifest)
    if labels.empty:
        raise ValueError("No labeled training data; run `pvd train` seeds first.")
    try:
        test = resolve_test_set(config)
    except ValueError:
        test = None

    specs = specs or default_specs(config)
    rows = []
    from .backbones import build_backbone

    for spec in specs:
        # Free the previous backbone: the registry cache would otherwise keep
        # every model of the grid in memory at once.
        build_backbone.cache_clear()
        log.info("Ablating %s", spec.name)
        try:
            rows.append(_run_spec(config, spec, labels, test))
        except Exception as exc:  # noqa: BLE001 - record and continue
            log.error("Configuration %s failed: %s", spec.name, exc)
            rows.append({**_EMPTY_ROW, "name": spec.name, "error": f"{type(exc).__name__}: {exc}"})
    report = write_report(config, "ablation", render_ablation(rows))
    log.info("Ablation report: %s", report)
    return report
