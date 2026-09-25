"""Optional timm fine-tuning on the collected labels (M9, §9.14).

This is the stage that can go past the linear-probe ceiling. It is deliberately
optional and slower; the probe remains the fast default.

Augmentation policy (explained because it encodes a modelling choice):
- **used:** random resized crop, horizontal flip, mild colour jitter, occasional
  grayscale, occasional JPEG re-compression. Grayscale matters because B&W photos
  and sketches exist in both classes.
- **avoided/mild:** strong blur, strong noise and heavy downscaling. Those erase
  or fake exactly the texture cues (sensor noise, paper grain, brush strokes)
  that separate hyperrealistic drawings from photos (G24, G30).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from .config import Config
from .device import pick_device
from .evaluate import evaluate_bundle, evaluate_if_test_set, render_report, write_report
from .features import package_versions
from .imageio import load_rgb
from .labels import effective_labels, label_snapshot_sha
from .models import load_any, newest_bundle_path
from .probe import MIN_TRAIN_PER_CLASS
from .scan import load_manifest

log = logging.getLogger(__name__)


@dataclass
class FinetuneBundle:
    """Bundle metadata for a fine-tuned model (§8.4, ``kind='finetune'``)."""

    classes: list[str]
    model_name: str
    image_size: int
    checkpoint: str
    mean: list[float]
    std: list[float]
    val_balanced_accuracy: float | None = None
    test_metrics: dict | None = None
    n_train: int = 0
    label_snapshot_sha: str = ""
    config: dict = field(default_factory=dict)
    created: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    versions: dict = field(default_factory=dict)
    kind: str = "finetune"


class _Dataset(Dataset):
    def __init__(self, paths: list[str], labels: np.ndarray, transform, bg, max_pixels: int) -> None:
        self.paths = paths
        self.labels = labels
        self.transform = transform
        self.bg = bg
        self.max_pixels = max_pixels

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        image = load_rgb(Path(self.paths[index]), self.bg, self.max_pixels)
        return self.transform(image), int(self.labels[index])


def _build_model(model_name: str, num_classes: int, pretrained: bool):
    """A timm model with a fresh ``num_classes`` head."""
    import timm

    return timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)


def _transforms(model, image_size: int):
    """Train/eval transforms; see the module docstring for the augmentation policy."""
    import timm

    data_config = timm.data.resolve_model_data_config(model)
    mean = list(data_config["mean"])
    std = list(data_config["std"])
    train_tf = v2.Compose(
        [
            v2.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
            v2.RandomHorizontalFlip(),
            v2.ColorJitter(0.2, 0.2, 0.2, 0.05),
            v2.RandomApply([v2.Grayscale(num_output_channels=3)], p=0.1),
            v2.ToImage(),
            v2.RandomApply([v2.JPEG(quality=[30, 95])], p=0.3),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std),
        ]
    )
    return train_tf, _eval_transform(image_size, mean, std), mean, std


def _eval_transform(image_size: int, mean: list[float], std: list[float]):
    """Deterministic resize + center crop, shared by validation and prediction."""
    return v2.Compose(
        [
            v2.Resize(image_size),
            v2.CenterCrop(image_size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std),
        ]
    )


def _split_parameters(model):
    """Split parameters into (new classifier head, pretrained backbone).

    The head is found through timm's ``get_classifier()`` rather than by name
    matching: names like ``fc`` also occur inside every ConvNeXt/ViT block
    (``mlp.fc1``/``fc2``), which would put most of the backbone on the 10x
    higher head learning rate.
    """
    head_ids = {id(param) for param in model.get_classifier().parameters()}
    head, backbone = [], []
    for param in model.parameters():
        (head if id(param) in head_ids else backbone).append(param)
    return head, backbone


@torch.inference_mode()
def finetune_predict_proba(
    bundle: FinetuneBundle, paths: list[str], config: Config
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(probs, valid)`` for image paths with a fine-tuned bundle.

    Undecodable images get ``valid=False`` and NaN probabilities instead of
    aborting the whole run.
    """
    device = pick_device(config.device)
    model = _build_model(bundle.model_name, len(bundle.classes), pretrained=False)
    checkpoint = torch.load(config.resolve(bundle.checkpoint), map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval().to(device)

    eval_tf = _eval_transform(bundle.image_size, bundle.mean, bundle.std)
    bg = tuple(config.image.alpha_background)
    probs = np.full((len(paths), len(bundle.classes)), np.nan, dtype=np.float64)
    valid = np.zeros(len(paths), dtype=bool)
    batch_size = config.finetune.batch_size
    for start in range(0, len(paths), batch_size):
        tensors, rows = [], []
        for offset, path in enumerate(paths[start : start + batch_size]):
            try:
                tensors.append(eval_tf(load_rgb(Path(path), bg, config.image.max_pixels)))
                rows.append(start + offset)
            except Exception as exc:  # noqa: BLE001 - one bad file must not stop the run
                log.error("Could not load %s: %s", path, exc)
        if not tensors:
            continue
        logits = model(torch.stack(tensors).to(device))
        probs[rows] = torch.softmax(logits.float(), dim=1).cpu().numpy()
        valid[rows] = True
    return probs, valid


def run_finetune(config: Config) -> Path:
    """Fine-tune a timm model, save a bundle + checkpoint, and write a report."""
    manifest = load_manifest(config)
    labels = effective_labels(config, manifest)
    if labels.empty:
        raise ValueError("No labeled training data; seed or review images first.")

    class_to_index = {name: i for i, name in enumerate(config.classes)}
    counts = labels["label"].value_counts()
    small = [name for name in config.classes if counts.get(name, 0) < MIN_TRAIN_PER_CLASS]
    if small:
        raise ValueError(f"Not enough labeled examples for {small}; need >= {MIN_TRAIN_PER_CLASS} each.")

    paths = [str(config.resolve(p)) for p in labels["path"].tolist()]
    y = np.array([class_to_index[label] for label in labels["label"]], dtype=int)
    # A stratified split needs at least one validation image per class.
    n_val = max(int(round(config.finetune.val_fraction * len(y))), len(config.classes))
    if len(y) - n_val < len(config.classes):
        raise ValueError(
            f"Too few labeled images ({len(y)}) for a stratified train/val split over "
            f"{len(config.classes)} classes; label more images first."
        )
    train_paths, val_paths, y_train, y_val = train_test_split(
        paths,
        y,
        test_size=n_val,
        stratify=y,
        random_state=config.seed,
    )

    device = pick_device(config.device)
    model = _build_model(config.finetune.model_name, len(config.classes), config.finetune.pretrained)
    model.to(device)
    train_tf, eval_tf, mean, std = _transforms(model, config.finetune.image_size)

    bg = tuple(config.image.alpha_background)
    max_pixels = config.image.max_pixels
    train_loader = DataLoader(
        _Dataset(train_paths, y_train, train_tf, bg, max_pixels),
        batch_size=config.finetune.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        # A trailing batch of 1 breaks BatchNorm in train mode.
        drop_last=len(train_paths) % config.finetune.batch_size == 1,
    )
    val_loader = DataLoader(
        _Dataset(val_paths, y_val, eval_tf, bg, max_pixels),
        batch_size=config.finetune.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    head, backbone = _split_parameters(model)
    optimizer = torch.optim.AdamW(
        [
            {"params": head, "lr": config.finetune.lr_head},
            {"params": backbone, "lr": config.finetune.lr_backbone},
        ],
        weight_decay=config.finetune.weight_decay,
    )
    steps_per_epoch = max(1, len(train_loader))
    warmup = max(1, int(0.1 * steps_per_epoch * config.finetune.epochs))
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, steps_per_epoch * config.finetune.epochs - warmup)),
        ],
        milestones=[warmup],
    )

    class_counts = np.bincount(y_train, minlength=len(config.classes)).clip(min=1)
    weights = torch.tensor(len(y_train) / (len(config.classes) * class_counts), dtype=torch.float32, device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=config.finetune.label_smoothing)

    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_state, best_ba, patience = None, -1.0, 0
    for epoch in range(1, config.finetune.epochs + 1):
        model.train()
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = criterion(model(images), targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        val_ba = _validate(model, val_loader, device)
        log.info("Epoch %d/%d: val balanced accuracy=%.3f", epoch, config.finetune.epochs, val_ba)
        if val_ba > best_ba:
            best_ba, patience = val_ba, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= config.finetune.early_stopping_patience:
                log.info("Early stopping at epoch %d", epoch)
                break

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    config.models_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = config.models_dir / f"finetune_{stamp}.pt"
    torch.save({"state_dict": best_state or model.state_dict(), "model_name": config.finetune.model_name}, checkpoint_path)

    bundle = FinetuneBundle(
        classes=list(config.classes),
        model_name=config.finetune.model_name,
        image_size=config.finetune.image_size,
        checkpoint=config.relative(checkpoint_path),
        mean=mean,
        std=std,
        val_balanced_accuracy=float(best_ba),
        n_train=int(len(y_train)),
        label_snapshot_sha=label_snapshot_sha(labels),
        config=config.to_dict(),
        versions=package_versions(),
    )

    result = evaluate_if_test_set(config, bundle)
    lines = [
        f"- Model: `{bundle.model_name}` @ {bundle.image_size}px",
        f"- Training examples: **{bundle.n_train}** (val fraction {config.finetune.val_fraction})",
        f"- Best val balanced accuracy: **{best_ba:.3f}**",
        f"- Label snapshot: `{bundle.label_snapshot_sha[:12]}`",
    ]
    if result is not None:
        bundle.test_metrics = {
            "accuracy": result.accuracy,
            "balanced_accuracy": result.balanced_accuracy,
            "auc": result.auc,
            "per_tag": result.per_tag,
        }
        lines.insert(3, f"- Test balanced accuracy: **{result.balanced_accuracy:.3f}**")
    lines = lines + [""] + _compare_with_probe(config, result)

    path = config.models_dir / f"finetune_{stamp}.joblib"
    joblib.dump(bundle, path)
    report = write_report(
        config,
        "finetune",
        render_report(f"finetune:{bundle.model_name}", result, lines) if result else "# Fine-tune report\n\n" + "\n".join(lines),
    )
    log.info("Saved fine-tuned bundle: %s (report %s)", path, report)
    return path


@torch.inference_mode()
def _validate(model, loader, device: str) -> float:
    model.eval()
    truth, preds = [], []
    for images, targets in loader:
        logits = model(images.to(device))
        preds.extend(logits.argmax(dim=1).cpu().tolist())
        truth.extend(targets.tolist())
    if not truth:
        return 0.0
    return float(balanced_accuracy_score(truth, preds))


def _compare_with_probe(config: Config, result) -> list[str]:
    """Re-score the newest probe on *today's* test set, next to the fine-tune.

    Stored probe metrics may come from an older test set, so the probe is
    evaluated again here; the table is per tag, where the differences matter.
    """
    if result is None:
        return ["No test set, so no probe comparison.", ""]
    try:
        path = newest_bundle_path(config, ("probe",))
        probe_result = evaluate_bundle(config, load_any(path))
    except (FileNotFoundError, ValueError) as exc:
        return [f"No probe to compare against ({exc}).", ""]

    lines = [
        "## Probe vs fine-tune (same test set, evaluated now)",
        "",
        "| slice | probe | fine-tune |",
        "|---|---|---|",
        f"| overall balanced accuracy | {probe_result.balanced_accuracy:.3f} | {result.balanced_accuracy:.3f} |",
    ]
    for tag in sorted(set(probe_result.per_tag) | set(result.per_tag)):
        probe_acc = probe_result.per_tag.get(tag, {}).get("accuracy")
        fine_acc = result.per_tag.get(tag, {}).get("accuracy")
        fmt = lambda v: "n/a" if v is None else f"{v:.3f}"  # noqa: E731
        lines.append(f"| tag `{tag}` accuracy | {fmt(probe_acc)} | {fmt(fine_acc)} |")
    lines += ["", f"Probe bundle: `{path.name}`", ""]
    return lines
