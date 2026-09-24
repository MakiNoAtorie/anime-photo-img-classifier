"""Gradio active-learning review app (§10).

The interactive behaviour lives in :class:`ReviewSession`, a plain Python object
with no Gradio dependency. That keeps the state machine unit-testable (select,
set label, confirm, undo) without a browser, and makes the Gradio code a thin
view layer.

Design rules implemented here:
- clicking a thumbnail only *selects* it; it never changes a label (G18),
- labels are set with one button per class plus ``skip``, so any N works (G20),
- confirming records unchanged images as "model was right",
- confirming retrains on cached embeddings and resamples in a few seconds (NF2).
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .evaluate import evaluate_arrays, resolve_test_set
from .features import get_features, load_embedding_cache
from .imageio import load_rgb
from .labels import LabelEvent, LabelStore, effective_labels
from .probe import train_probe
from .sampler import next_batch
from .scan import load_manifest

log = logging.getLogger(__name__)

THUMBNAIL_SIZE = 384


# --------------------------------------------------------------------------- #
# Session
# --------------------------------------------------------------------------- #
@dataclass
class SessionItem:
    sha1: str
    path_rel: str
    path_abs: str
    sampled_as: str
    model_pred: str
    model_conf: float


def make_thumbnail(config: Config, sha1: str, source: Path) -> Path:
    """Create (once) and return a cached 384px JPEG thumbnail (G18, note 9)."""
    target = config.thumbs_dir / f"{sha1}.jpg"
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    image = load_rgb(source, tuple(config.image.alpha_background), config.image.max_pixels)
    image.thumbnail((THUMBNAIL_SIZE, THUMBNAIL_SIZE))
    image.convert("RGB").save(target, format="JPEG", quality=85)
    return target


class ReviewSession:
    """Holds all mutable review state. Single-user by assumption (§10.3)."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.manifest = load_manifest(config)
        self.store = LabelStore(config.labels_file)
        self.class_to_index = {name: i for i, name in enumerate(config.classes)}

        self.ok = self.manifest[self.manifest["status"] == "ok"].reset_index(drop=True)
        self.sha_to_pos = {sha: pos for pos, sha in enumerate(self.ok["sha1"])}
        self.features = self._load_features()
        self._backbone = None
        self.backbone_dim = self._global_dim()

        self.model = None
        self.best_C = 10.0
        self.model_kind = "none"
        self.cv_balanced_accuracy: float | None = None
        self.last_test_ba: float | None = None
        self.delta_test_ba: float | None = None
        self.last_batch_summary: dict | None = None

        self.batch: list[SessionItem] = []
        self.pending: dict[int, str] = {}
        self.selected: int | None = None
        self._fit_model()
        self._recompute_scores()
        self.next_batch()

    # -- features / model ---------------------------------------------------
    def _load_features(self) -> np.ndarray:
        if self.ok.empty:
            return np.zeros((0, 0), dtype=np.float32)
        shas = self.ok["sha1"].tolist()
        paths = [str(self.config.resolve(p)) for p in self.ok["path"].tolist()]
        return get_features(self.config, shas, paths)

    def _global_dim(self) -> int:
        key = self.config.backbone.id
        try:
            _, _, meta = load_embedding_cache(self.config, key)
            return int(meta.get("block_sizes", {}).get("global", self.features.shape[1]))
        except Exception:  # noqa: BLE001
            return self.features.shape[1]

    def _training_rows(self) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        labels = effective_labels(self.config, self.manifest)
        positions, targets = [], []
        for _, row in labels.iterrows():
            pos = self.sha_to_pos.get(row["sha1"])
            if pos is None:
                continue
            positions.append(pos)
            targets.append(self.class_to_index[row["label"]])
        X = self.features[positions] if positions else np.zeros((0, self.features.shape[1]), np.float32)
        y = np.array(targets, dtype=int)
        return X, y, labels

    def _fit_model(self) -> None:
        X, y, _ = self._training_rows()
        counts = np.bincount(y, minlength=len(self.config.classes)) if len(y) else np.zeros(len(self.config.classes), int)
        if len(y) and counts.min() >= 2:
            bundle = train_probe(
                X,
                y,
                self.config,
                classes=list(self.config.classes),
                backbone_id=self.config.backbone.id,
            )
            self.model = bundle.estimator
            self.best_C = bundle.best_C or self.best_C
            self.model_kind = "probe"
            self.cv_balanced_accuracy = bundle.cv_balanced_accuracy
        elif self._has_text_backbone():
            self.model_kind = "zeroshot"
        else:
            self.model_kind = "none"

    def _has_text_backbone(self) -> bool:
        try:
            from .backbones import build_backbone
            from .device import pick_device

            backbone = build_backbone(self.config.backbone.id, pick_device(self.config.device))
            return bool(getattr(backbone, "is_text_aligned", False))
        except Exception:  # noqa: BLE001
            return False

    def _get_backbone(self):
        if self._backbone is None:
            from .backbones import build_backbone
            from .device import pick_device

            self._backbone = build_backbone(self.config.backbone.id, pick_device(self.config.device))
        return self._backbone

    def _recompute_scores(self) -> None:
        n = len(self.ok)
        n_classes = len(self.config.classes)
        if n == 0 or self.features.size == 0:
            self.probs = np.zeros((n, n_classes), dtype=np.float64)
            return
        if self.model_kind == "probe":
            self.probs = self.model.predict_proba(self.features)
        elif self.model_kind == "zeroshot":
            from .zeroshot import class_text_centroids, softmax

            backbone = self._get_backbone()
            global_features = self.features[:, : self.backbone_dim]
            centroids = class_text_centroids(backbone, self.config.zeroshot.prompts, list(self.config.classes))
            scale = getattr(backbone, "logit_scale", None) or 100.0
            self.probs = softmax((global_features @ centroids.T) * scale)
        else:
            self.probs = np.full((n, n_classes), 1.0 / n_classes)

    # -- candidates / batches ----------------------------------------------
    def labeled_shas(self) -> set[str]:
        return set(self.store.current())

    def candidate_positions(self) -> list[int]:
        labeled = self.labeled_shas()
        is_pool = (self.ok["split"] == "pool").to_numpy()
        return [pos for pos in range(len(self.ok)) if is_pool[pos] and self.ok["sha1"][pos] not in labeled]

    def next_batch(self) -> list[SessionItem]:
        candidates = self.candidate_positions()
        picks = next_batch(self.probs, candidates, self.features, self.config, self.rng)
        self.batch = []
        for pos, sampled_as in picks:
            self.batch.append(self._make_item(pos, sampled_as))
        self.pending = {}
        return self.batch

    def load_audit_batch(self, limit: int | None = None) -> list[SessionItem]:
        """Replace the batch with likely label mistakes for re-review (G21)."""
        from .audit import label_error_suspects

        suspects = label_error_suspects(self.config, self.manifest)
        if limit is not None:
            suspects = suspects[:limit]
        self.batch = []
        for suspect in suspects:
            position = self.sha_to_pos.get(suspect["sha1"])
            if position is None:
                continue
            self.batch.append(self._make_item(position, "audit"))
        self.pending = {}
        return self.batch

    def _make_item(self, pos: int, sampled_as: str) -> SessionItem:
        row = self.ok.iloc[pos]
        pred_index = int(self.probs[pos].argmax())
        return SessionItem(
            sha1=row["sha1"],
            path_rel=row["path"],
            path_abs=str(self.config.resolve(row["path"])),
            sampled_as=sampled_as,
            model_pred=self.config.classes[pred_index],
            model_conf=float(self.probs[pos].max()),
        )

    def select(self, position: int) -> SessionItem | None:
        if 0 <= position < len(self.batch):
            return self.batch[position]
        return None

    def set_label(self, position: int, label: str) -> None:
        if 0 <= position < len(self.batch):
            self.pending[position] = label

    def current_label(self, position: int) -> str | None:
        if position in self.pending:
            return self.pending[position]
        return self.batch[position].model_pred

    def changed(self, position: int) -> bool:
        return position in self.pending and self.pending[position] != self.batch[position].model_pred

    def path_of(self, position: int) -> Path:
        return Path(self.batch[position].path_abs)

    # -- confirm / undo -----------------------------------------------------
    def confirm(self) -> dict:
        events: list[LabelEvent] = []
        wrong: list[bool] = []
        random_wrong: list[bool] = []
        for position, item in enumerate(self.batch):
            explicit = self.pending.get(position)
            if explicit is None:
                if self.model_kind == "none":
                    continue  # no prediction to trust; leave unlabeled
                final = item.model_pred
            else:
                final = explicit
            events.append(
                LabelEvent(
                    sha1=item.sha1,
                    label=final,
                    path=item.path_rel,
                    prev=None,
                    source="audit_fix" if item.sampled_as == "audit" else "review",
                    model_pred=item.model_pred,
                    model_conf=item.model_conf,
                    sampled_as=item.sampled_as,
                )
            )
            is_wrong = final != item.model_pred
            wrong.append(is_wrong)
            if item.sampled_as == "random":
                random_wrong.append(is_wrong)

        batch_id = self.store.add_batch(events) if events else None
        changed = sum(1 for position in self.pending if self.changed(position))
        summary = {
            "batch_id": batch_id,
            "confirmed": len(events),
            "changed": changed,
            "batch_error_rate": float(np.mean(wrong)) if wrong else 0.0,
            "random_error_rate": float(np.mean(random_wrong)) if random_wrong else None,
            "random_n": len(random_wrong),
        }
        self.last_batch_summary = summary
        self._refresh_after_label_change()
        return summary

    def undo(self) -> str | None:
        undone = self.store.undo_last()
        if undone is not None:
            self._refresh_after_label_change()
        return undone

    def _refresh_after_label_change(self) -> None:
        previous = self.last_test_ba
        self._fit_model()
        self._recompute_scores()
        current = self.evaluate_test()
        self.last_test_ba = current
        if previous is not None and current is not None:
            self.delta_test_ba = current - previous
        self.next_batch()

    def evaluate_test(self) -> float | None:
        """Balanced accuracy on the held-out test set, or None if unavailable."""
        if self.model_kind != "probe":
            return None
        test_mask = (self.ok["split"] == "test").to_numpy()
        if not test_mask.any():
            return None
        try:
            test = resolve_test_set(self.config)
        except ValueError:
            return None
        positions = [self.sha_to_pos[sha] for sha in test.shas if sha in self.sha_to_pos]
        if len(positions) != len(test.shas):
            return None
        X = self.features[positions]
        result = evaluate_arrays(
            test.y_true, self.model.predict_proba(X), list(self.config.classes), test.frame, test.tags
        )
        return result.balanced_accuracy

    # -- status / value helpers --------------------------------------------
    def labeled_counts(self) -> dict[str, int]:
        counts = {name: 0 for name in self.config.classes}
        for label in self.store.current().values():
            if label in counts:
                counts[label] += 1
        return counts

    def batch_labels(self) -> list[str]:
        return [self.current_label(i) or "?" for i in range(len(self.batch))]

    def status_markdown(self) -> str:
        counts = self.labeled_counts()
        count_text = ", ".join(f"{name}={counts[name]}" for name in self.config.classes)
        ba = f"{self.last_test_ba * 100:.2f}%" if self.last_test_ba is not None else "n/a"
        delta = ""
        if self.delta_test_ba is not None:
            delta = f" ({self.delta_test_ba * 100:+.2f})"
        lines = [
            f"**Batch {self.store.next_batch_id()}** · {len(self.batch)} shown · "
            f"labeled: {count_text} · model: `{self.model_kind}`"
        ]
        lines.append(f"Test balanced accuracy: **{ba}{delta}**")
        if self.last_batch_summary:
            s = self.last_batch_summary
            random_text = (
                f"{s['random_error_rate'] * 100:.0f}% on {s['random_n']} random"
                if s["random_error_rate"] is not None
                else "no random items"
            )
            lines.append(
                f"Last batch: changed {s['changed']}/{s['confirmed']} · "
                f"error {s['batch_error_rate'] * 100:.0f}% overall, {random_text} "
                "(the random slice is the unbiased one — hard/uncertain items are chosen on purpose)"
            )
        if not self.batch:
            lines.append("")
            lines.append("**All pool images reviewed.** Run `pvd train` and `pvd eval` for the final report.")
        return "\n\n".join(lines)


# --------------------------------------------------------------------------- #
# Gradio view
# --------------------------------------------------------------------------- #
def _contact_sheet(session: ReviewSession) -> list[tuple[str, str]]:
    items = []
    for position, item in enumerate(session.batch):
        thumb = make_thumbnail(session.config, item.sha1, Path(item.path_abs))
        marker = " *" if session.changed(position) else ""
        caption = f"{session.current_label(position)}{marker}"
        items.append((str(thumb), caption))
    return items


def build_app(session: ReviewSession):
    """Wire the session into a Gradio Blocks app (does not launch)."""
    import gradio as gr

    # Quick mode is a tiny piece of view state; selection lives on the session.
    view_state = {"quick": False}

    def view(position: int | None):
        gallery = _contact_sheet(session)
        if position is None or not (0 <= position < len(session.batch)):
            return gallery, None, "_Select an image to inspect it._", session.status_markdown()
        item = session.batch[position]
        detail = (
            f"**{Path(item.path_rel).name}**\n\n"
            f"Model: `{item.model_pred}` ({item.model_conf:.2f}) · "
            f"Current: `{session.current_label(position)}`"
            + (" **(changed)**" if session.changed(position) else "")
            + f"\n\nSampled as: `{item.sampled_as}`"
        )
        return gallery, item.path_abs, detail, session.status_markdown()

    with gr.Blocks(title="Photo vs Drawing review") as demo:
        gr.Markdown("## Photo vs Drawing — review")
        status = gr.Markdown(session.status_markdown())
        label_buttons: list[tuple[str, gr.Button]] = []

        with gr.Row():
            with gr.Column(scale=3):
                gallery = gr.Gallery(
                    value=_contact_sheet(session),
                    columns=6,
                    rows=4,
                    height="auto",
                    allow_preview=False,
                    label="Batch (click to select)",
                )
            with gr.Column(scale=2):
                preview = gr.Image(label="Preview (full resolution)", type="filepath", height=420)
                detail = gr.Markdown("_Select an image to inspect it._")
                with gr.Row():
                    for label in list(session.config.classes) + [session.config.skip_label]:
                        label_buttons.append((label, gr.Button(label)))

        with gr.Row():
            confirm_btn = gr.Button("Confirm & retrain", variant="primary")
            undo_btn = gr.Button("Undo")
            audit_btn = gr.Button("Audit mode")
            quick_toggle = gr.Checkbox(label="Quick mode (click cycles labels)", value=False)

        outputs = [gallery, preview, detail, status]

        def on_select(evt: gr.SelectData):
            raw = evt.index
            position = int(raw[0]) if isinstance(raw, (list, tuple)) else int(raw)
            if view_state["quick"]:
                cycle = list(session.config.classes) + [session.config.skip_label]
                current = session.current_label(position)
                next_label = cycle[(cycle.index(current) + 1) % len(cycle)] if current in cycle else cycle[0]
                session.set_label(position, next_label)
            session.selected = position
            return view(position)

        gallery.select(on_select, inputs=None, outputs=outputs)

        def make_set_label(label: str):
            def set_label_fn():
                position = session.selected
                if position is not None:
                    session.set_label(position, label)
                return view(position)

            return set_label_fn

        # One button per class (+ skip) so the UI works for any N (G20).
        for label, button in label_buttons:
            button.click(make_set_label(label), inputs=None, outputs=outputs)

        def on_quick(flag: bool):
            view_state["quick"] = flag
            return flag

        quick_toggle.change(on_quick, inputs=[quick_toggle], outputs=[quick_toggle])

        def on_confirm():
            session.confirm()
            session.selected = None
            return view(None)

        def on_undo():
            session.undo()
            session.selected = None
            return view(None)

        def on_audit():
            session.load_audit_batch()
            session.selected = None
            return view(None)

        confirm_btn.click(on_confirm, inputs=None, outputs=outputs)
        undo_btn.click(on_undo, inputs=None, outputs=outputs)
        audit_btn.click(on_audit, inputs=None, outputs=outputs)

    return demo


def run_review(config: Config) -> None:
    """Build the session and launch the app (called from the CLI)."""
    from .scan import load_manifest  # noqa: F401 - fail clearly if no manifest

    session = ReviewSession(config)
    demo = build_app(session)
    demo.launch()
