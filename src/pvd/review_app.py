"""Gradio active-learning review app (§10).

The interactive behaviour lives in :class:`ReviewSession`, a plain Python object
with no Gradio dependency. That keeps the state machine unit-testable (select,
set label, cycle, confirm, undo, modes) without a browser, and makes the Gradio
code a thin view layer.

Design rules implemented here:
- clicking a thumbnail only *selects* it; it never changes a label (G18),
- labels are set with one button per class plus ``skip``, so any N works (G20),
- in active/random mode, confirming records unchanged images as "model was
  right"; in audit mode, unchanged images keep their **stored** label (only
  explicit corrections are written as ``audit_fix``),
- confirming refits the probe on cached embeddings with the last chosen ``C``
  and resamples in a few seconds (NF2); a full cross-validated retrain runs
  every ``review.cv_every`` batches,
- until every class has ``MIN_TRAIN_PER_CLASS`` labels the app uses zero-shot
  (or no model) and says why.

Single-user by assumption: one session object per process (§10.3).

Note: this module intentionally does **not** use ``from __future__ import
annotations``. Gradio inspects the ``gr.SelectData`` type hint at runtime to
decide whether to pass the click event; a string annotation would hide it.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .evaluate import evaluate_arrays, resolve_test_set
from .features import get_features, global_block, load_feature_meta
from .imageio import load_rgb
from .labels import LabelEvent, LabelStore, effective_labels
from .probe import MIN_TRAIN_PER_CLASS, quick_retrain, train_probe
from .sampler import next_batch
from .scan import load_manifest

log = logging.getLogger(__name__)

THUMBNAIL_SIZE = 384
MODES = ("active", "random", "audit")


# --------------------------------------------------------------------------- #
# Session
# --------------------------------------------------------------------------- #
@dataclass
class SessionItem:
    """One image in the current batch."""

    sha1: str
    path_rel: str
    path_abs: str
    sampled_as: str
    model_pred: str
    model_conf: float
    #: Label saved if the reviewer does not touch the item: the model's guess in
    #: active/random mode, the stored label in audit mode.
    default_label: str
    #: The label this image already has (seed folder or log), if any.
    stored_label: str | None = None


def make_thumbnail(config: Config, sha1: str, source: Path) -> Path:
    """Create (once) and return a cached 384px JPEG thumbnail (G18, note 9).

    Falls back to the original file if it cannot be decoded, so one bad file
    never breaks the gallery.
    """
    target = config.thumbs_dir / f"{sha1}.jpg"
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        image = load_rgb(source, tuple(config.image.alpha_background), config.image.max_pixels)
        image.thumbnail((THUMBNAIL_SIZE, THUMBNAIL_SIZE))
        image.convert("RGB").save(target, format="JPEG", quality=85)
    except Exception as exc:  # noqa: BLE001
        log.warning("Thumbnail failed for %s: %s", source, exc)
        return source
    return target


class ReviewSession:
    """Holds all mutable review state. Single-user by assumption (§10.3)."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.manifest = load_manifest(config)
        self.store = LabelStore(config.labels_file)
        self.class_to_index = {name: i for i, name in enumerate(config.classes)}

        ok = self.manifest[self.manifest["status"] == "ok"].reset_index(drop=True)
        self.features, self.ok = self._load_features(ok)
        self.sha_to_pos = {sha: pos for pos, sha in enumerate(self.ok["sha1"])}
        self.meta = load_feature_meta(config)
        # Unit-norm, text-aligned block: used for zero-shot and similarity.
        self.global_features = global_block(self.features, self.meta) if self.features.size else self.features
        self._backbone = None
        self._centroids: np.ndarray | None = None

        self.model = None
        self.best_C: float | None = None
        self.model_kind = "none"
        self.model_reason = ""
        self.batches_since_cv = 0
        self.cv_balanced_accuracy: float | None = None
        self.last_test_ba: float | None = None
        self.delta_test_ba: float | None = None
        self.last_batch_summary: dict | None = None
        self.last_action = ""

        self.mode = "active"
        self.batch: list[SessionItem] = []
        self.pending: dict[int, str] = {}
        self.selected: int | None = None
        self._fit_model(full_cv=True)
        self._recompute_scores()
        self.last_test_ba = self.evaluate_test()
        self.next_batch()

    # -- features / model ---------------------------------------------------
    def _load_features(self, ok: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
        """Feature matrix for ``ok`` rows, dropping images that failed to embed."""
        if ok.empty:
            return np.zeros((0, 0), dtype=np.float32), ok
        shas = ok["sha1"].tolist()
        paths = [str(self.config.resolve(p)) for p in ok["path"].tolist()]
        features, valid = get_features(self.config, shas, paths, return_mask=True)
        if not valid.all():
            ok = ok[valid].reset_index(drop=True)
            features = features[valid]
        return features, ok

    def _labels(self) -> pd.DataFrame:
        return effective_labels(self.config, self.manifest, self.store)

    def _training_rows(self) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        labels = self._labels()
        positions = labels["sha1"].map(self.sha_to_pos)
        labels = labels[positions.notna()].reset_index(drop=True)
        positions = positions.dropna().astype(int).to_numpy()
        X = self.features[positions] if len(positions) else np.zeros((0, self.features.shape[1]), np.float32)
        y = labels["label"].map(self.class_to_index).to_numpy(dtype=int)
        return X, y, labels

    def _fit_model(self, full_cv: bool = False) -> None:
        """Pick probe / zero-shot / none and (re)fit the probe.

        The probe needs ``MIN_TRAIN_PER_CLASS`` labels in every class (§9.8).
        Full cross-validation runs at startup and every ``review.cv_every``
        batches; otherwise the probe is refit once with the last chosen ``C``.
        """
        X, y, _ = self._training_rows()
        n_classes = len(self.config.classes)
        counts = np.bincount(y, minlength=n_classes) if len(y) else np.zeros(n_classes, int)
        if counts.min() >= MIN_TRAIN_PER_CLASS:
            if full_cv or self.model is None or self.best_C is None:
                bundle = train_probe(
                    X, y, self.config, classes=list(self.config.classes), backbone_id=self.config.backbone.id
                )
                self.model = bundle.estimator
                self.best_C = bundle.best_C
                self.cv_balanced_accuracy = bundle.cv_balanced_accuracy
                self.batches_since_cv = 0
            else:
                self.model = quick_retrain(X, y, self.best_C, self.config)
            self.model_kind = "probe"
            self.model_reason = f"linear probe on {len(y)} labels (C={self.best_C:g})"
            return

        self.model = None
        have = ", ".join(f"{name}={int(n)}" for name, n in zip(self.config.classes, counts))
        need = f"the probe needs {MIN_TRAIN_PER_CLASS} labels per class (have {have})"
        text_ok, why_not = self._zeroshot_available()
        if text_ok:
            self.model_kind = "zeroshot"
            self.model_reason = f"zero-shot text prompts until {need}"
        else:
            self.model_kind = "none"
            self.model_reason = f"no predictions yet: {why_not}, and {need}"

    def _zeroshot_available(self) -> tuple[bool, str]:
        from .zeroshot import missing_prompts

        missing = missing_prompts(self.config.zeroshot.prompts, list(self.config.classes))
        if missing:
            return False, f"no zero-shot prompts for {missing}"
        try:
            backbone = self._get_backbone()
        except Exception as exc:  # noqa: BLE001
            return False, f"backbone unavailable ({exc})"
        if not getattr(backbone, "is_text_aligned", False):
            return False, f"backbone {self.config.backbone.id!r} has no text encoder"
        return True, ""

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
            if self._centroids is None:
                self._centroids = class_text_centroids(
                    backbone, self.config.zeroshot.prompts, list(self.config.classes)
                )
            scale = getattr(backbone, "logit_scale", None) or 100.0
            self.probs = softmax((self.global_features @ self._centroids.T) * scale)
        else:
            self.probs = np.full((n, n_classes), 1.0 / n_classes)

    # -- candidates / batches ----------------------------------------------
    def labeled_shas(self) -> set[str]:
        """Pool images already decided (a class or skip) and so not candidates.

        Labels for classes that no longer exist (after a rename) don't count,
        so those images come back to review instead of vanishing.
        """
        valid = set(self.config.classes) | {self.config.skip_label}
        return {sha for sha, label in self.store.current().items() if label in valid}

    def candidate_positions(self) -> list[int]:
        is_pool = (self.ok["split"] == "pool").to_numpy()
        labeled = self.ok["sha1"].isin(self.labeled_shas()).to_numpy()
        return np.flatnonzero(is_pool & ~labeled).tolist()

    def set_mode(self, mode: str) -> list[SessionItem]:
        """Switch between active learning, random sampling and audit batches."""
        if mode not in MODES:
            raise ValueError(f"Unknown mode {mode!r}; use one of {MODES}")
        self.mode = mode
        return self.next_batch()

    def next_batch(self) -> list[SessionItem]:
        """Load the next batch for the current mode and clear pending edits."""
        if self.mode == "audit":
            return self.load_audit_batch()
        candidates = self.candidate_positions()
        if self.mode == "random":
            size = min(self.config.review.batch_size, len(candidates))
            chosen = self.rng.choice(candidates, size=size, replace=False) if size else []
            picks = [(int(pos), "random") for pos in chosen]
        else:
            picks = next_batch(self.probs, candidates, self.global_features, self.config, self.rng)
        self.batch = [self._make_item(pos, sampled_as) for pos, sampled_as in picks]
        self.pending = {}
        self.selected = None
        return self.batch

    def load_audit_batch(self, limit: int | None = None) -> list[SessionItem]:
        """Replace the batch with likely label mistakes for re-review (G21).

        Unchanged audit items keep their stored label on confirm; only explicit
        corrections are written (as ``audit_fix``).
        """
        from .audit import label_error_suspects

        self.mode = "audit"
        limit = limit if limit is not None else self.config.review.batch_size
        suspects = label_error_suspects(self.config, self.manifest, store=self.store)[:limit]
        self.batch = []
        for suspect in suspects:
            position = self.sha_to_pos.get(suspect["sha1"])
            if position is None:
                continue
            self.batch.append(self._make_item(position, "audit", stored_label=suspect["label"]))
        self.pending = {}
        self.selected = None
        return self.batch

    def _stored_label(self, sha1: str, row: pd.Series) -> str | None:
        logged = self.store.current().get(sha1)
        if logged is not None:
            return logged
        return row["folder_label"] if row["split"] == "seed" else None

    def _make_item(self, pos: int, sampled_as: str, stored_label: str | None = None) -> SessionItem:
        row = self.ok.iloc[pos]
        pred_index = int(self.probs[pos].argmax())
        model_pred = self.config.classes[pred_index]
        stored = stored_label if stored_label is not None else self._stored_label(row["sha1"], row)
        return SessionItem(
            sha1=row["sha1"],
            path_rel=row["path"],
            path_abs=str(self.config.resolve(row["path"])),
            sampled_as=sampled_as,
            model_pred=model_pred,
            model_conf=float(self.probs[pos].max()),
            default_label=(stored if sampled_as == "audit" and stored else model_pred),
            stored_label=stored,
        )

    # -- selection and labels ----------------------------------------------
    def select(self, position: int) -> SessionItem | None:
        """Select an item for the preview. Never changes a label (G18)."""
        if 0 <= position < len(self.batch):
            self.selected = position
            return self.batch[position]
        return None

    def move_selection(self, step: int) -> int | None:
        """Select the next/previous item (arrow keys), wrapping around."""
        if not self.batch:
            return None
        start = -1 if self.selected is None and step > 0 else (self.selected or 0)
        self.selected = (start + step) % len(self.batch)
        return self.selected

    def set_label(self, position: int, label: str) -> None:
        if 0 <= position < len(self.batch):
            self.pending[position] = label

    def label_cycle(self, position: int) -> list[str]:
        """Quick-mode order: model guess -> other classes -> skip (G20)."""
        pred = self.batch[position].model_pred
        others = [name for name in self.config.classes if name != pred]
        return [pred, *others, self.config.skip_label]

    def cycle_label(self, position: int) -> str | None:
        """Advance the item's label one step along :meth:`label_cycle`."""
        if not 0 <= position < len(self.batch):
            return None
        cycle = self.label_cycle(position)
        current = self.current_label(position)
        next_label = cycle[(cycle.index(current) + 1) % len(cycle)] if current in cycle else cycle[0]
        self.set_label(position, next_label)
        return next_label

    def current_label(self, position: int) -> str | None:
        if position in self.pending:
            return self.pending[position]
        return self.batch[position].default_label

    def changed(self, position: int) -> bool:
        return position in self.pending and self.pending[position] != self.batch[position].default_label

    def path_of(self, position: int) -> Path:
        return Path(self.batch[position].path_abs)

    # -- confirm / undo -----------------------------------------------------
    def confirm(self) -> dict:
        """Write the batch's labels, retrain and load the next batch."""
        events: list[LabelEvent] = []
        wrong: list[bool] = []
        random_wrong: list[bool] = []
        for position, item in enumerate(self.batch):
            explicit = self.pending.get(position)
            if item.sampled_as == "audit":
                final = explicit if explicit is not None else item.default_label
                if final == item.stored_label:
                    continue  # confirmed as already labeled: nothing to write
            elif explicit is None and self.model_kind == "none":
                continue  # no prediction to trust; leave unlabeled
            else:
                final = explicit if explicit is not None else item.default_label
            events.append(
                LabelEvent(
                    sha1=item.sha1,
                    label=final,
                    path=item.path_rel,
                    prev=item.stored_label,
                    source="audit_fix" if item.sampled_as == "audit" else "review",
                    model_pred=item.model_pred,
                    model_conf=item.model_conf,
                    sampled_as=item.sampled_as,
                )
            )
            # Error rates only count real class decisions: `skip` means "out of
            # scope", not "the model was wrong".
            if item.sampled_as != "audit" and final != self.config.skip_label:
                is_wrong = final != item.model_pred
                wrong.append(is_wrong)
                if item.sampled_as == "random":
                    random_wrong.append(is_wrong)

        batch_id = self.store.add_batch(events) if events else None
        summary = {
            "batch_id": batch_id,
            "mode": self.mode,
            "confirmed": len(events),
            "shown": len(self.batch),
            "changed": sum(1 for position in self.pending if self.changed(position)),
            "batch_error_rate": float(np.mean(wrong)) if wrong else None,
            "random_error_rate": float(np.mean(random_wrong)) if random_wrong else None,
            "random_n": len(random_wrong),
            "random_wrong": int(sum(random_wrong)),
        }
        self.last_batch_summary = summary
        self.last_action = (
            f"Saved batch `{batch_id}` ({len(events)} labels)." if batch_id else "Nothing to save in this batch."
        )
        if batch_id:
            self.batches_since_cv += 1
        if self.mode == "audit":
            self.mode = "active"  # an audit pass is one batch; resume active learning
        if batch_id and self.config.review.retrain_every_batch:
            self._refresh_after_label_change()
        else:
            self.next_batch()
        return summary

    def undo(self) -> str | None:
        """Revert the last confirmed batch (up to ``review.undo_depth`` in a row)."""
        undone = self.store.undo_last(max_depth=self.config.review.undo_depth)
        if undone is None:
            self.last_action = "Nothing to undo (or the undo limit was reached)."
            return None
        self.last_batch_summary = None
        self.last_action = f"Undid batch `{undone}`; its images are back in the pool."
        self._refresh_after_label_change()
        return undone

    def _refresh_after_label_change(self) -> None:
        full_cv = self.batches_since_cv >= self.config.review.cv_every
        self._fit_model(full_cv=full_cv)
        self._recompute_scores()
        current = self.evaluate_test()
        if self.last_test_ba is not None and current is not None:
            self.delta_test_ba = current - self.last_test_ba
        else:
            self.delta_test_ba = None
        self.last_test_ba = current
        self.next_batch()

    def evaluate_test(self) -> float | None:
        """Balanced accuracy on the held-out test set, or None if unavailable."""
        if self.model_kind != "probe":
            return None
        try:
            test = resolve_test_set(self.config, self.manifest)
        except ValueError:
            return None
        present = np.array([sha in self.sha_to_pos for sha in test.shas], dtype=bool)
        if not present.any():
            return None
        test = test.subset(present)
        X = self.features[[self.sha_to_pos[sha] for sha in test.shas]]
        result = evaluate_arrays(
            test.y_true, self.model.predict_proba(X), list(self.config.classes), test.frame, test.tags
        )
        return result.balanced_accuracy

    # -- status / value helpers --------------------------------------------
    def labeled_counts(self) -> dict[str, int]:
        """Training labels per class (seed folders + review/audit decisions)."""
        counts = self._labels()["label"].value_counts()
        return {name: int(counts.get(name, 0)) for name in self.config.classes}

    def batch_labels(self) -> list[str]:
        return [self.current_label(i) or "?" for i in range(len(self.batch))]

    def unchanged_hint(self) -> str:
        """What Confirm does with images the reviewer did not touch."""
        if self.mode == "audit":
            return "Audit: untouched images **keep their current label**; only your changes are saved."
        if self.model_kind == "none":
            return "No model yet: only images you label explicitly are saved."
        return "⚠️ Untouched images are saved **as the model's guess** when you press Confirm — check each one."

    def status_markdown(self) -> str:
        counts = self.labeled_counts()
        count_text = ", ".join(f"{name}={counts[name]}" for name in self.config.classes)
        ba = f"{self.last_test_ba * 100:.1f}%" if self.last_test_ba is not None else "n/a"
        delta = f" ({self.delta_test_ba * 100:+.1f} pp)" if self.delta_test_ba is not None else ""
        lines = [
            f"**Mode: {self.mode}** · batch {self.store.next_batch_id()} · {len(self.batch)} shown · "
            f"training labels: {count_text}",
            f"Model: {self.model_reason}",
            f"Test balanced accuracy (the real progress meter): **{ba}{delta}**",
        ]
        s = self.last_batch_summary
        if s and s.get("batch_error_rate") is not None:
            random_text = (
                f"random part: {s['random_wrong']}/{s['random_n']} wrong"
                if s["random_n"]
                else "no random items"
            )
            lines.append(
                f"Last batch: changed {s['changed']}/{s['shown']} ({random_text}). "
                "_Uncertain images are picked because they are hard, so the whole-batch "
                "error stays high even as the model improves; the random part is the unbiased one._"
            )
        if self.last_action:
            lines.append(self.last_action)
        if not self.batch:
            lines.append("")
            lines.append(self._empty_message())
        return "\n\n".join(lines)

    def _empty_message(self) -> str:
        if self.mode == "audit":
            return "**No likely label mistakes found.** Switch back to *active* mode."
        if not (self.ok["split"] == "pool").any():
            return (
                "**No pool images found.** Put unlabeled images in `data/pool/`, then run "
                "`pvd scan` and `pvd embed` and restart the app."
            )
        return "**All pool images reviewed.** Run `pvd train` and `pvd eval` for the final report."


# --------------------------------------------------------------------------- #
# Gradio view
# --------------------------------------------------------------------------- #
# Keyboard shortcuts: 1..9 label, 0/s skip, arrows move, Enter confirms. They
# click the matching buttons, so the keyboard and mouse share one code path.
KEYBOARD_JS = """
<script>
document.addEventListener('keydown', (event) => {
  const target = event.target;
  if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable)) return;
  if (event.metaKey || event.ctrlKey || event.altKey) return;
  let id = null;
  if (/^[1-9]$/.test(event.key)) id = 'pvd-label-' + (parseInt(event.key, 10) - 1);
  else if (event.key === '0' || event.key === 's') id = 'pvd-skip';
  else if (event.key === 'ArrowRight') id = 'pvd-next';
  else if (event.key === 'ArrowLeft') id = 'pvd-prev';
  else if (event.key === 'Enter') id = 'pvd-confirm';
  if (!id) return;
  const element = document.getElementById(id);
  if (element) { event.preventDefault(); element.click(); }
});
</script>
"""


def _contact_sheet(session: ReviewSession) -> list[tuple[str, str]]:
    """Gallery items: thumbnail + short caption.

    Unchanged: ``label 0.83`` (the model's confidence). Changed: ``label ✱``;
    the model's guess stays visible in the detail panel. Kept short because
    captions are truncated under small thumbnails.
    """
    items = []
    for position, item in enumerate(session.batch):
        thumb = make_thumbnail(session.config, item.sha1, Path(item.path_abs))
        label = session.current_label(position)
        caption = f"{label} ✱" if session.changed(position) else f"{label} {item.model_conf:.2f}"
        items.append((str(thumb), caption))
    return items


def build_app(session: ReviewSession):
    """Wire the session into a Gradio Blocks app (does not launch)."""
    import gradio as gr

    view_state = {"quick": False}
    label_names = list(session.config.classes) + [session.config.skip_label]

    def detail_text(position: int | None) -> str:
        if position is None or not (0 <= position < len(session.batch)):
            return "_Click a thumbnail to inspect it at full resolution._"
        item = session.batch[position]
        stored = f" · stored label: `{item.stored_label}`" if item.stored_label else ""
        return (
            f"**{Path(item.path_rel).name}** ({position + 1}/{len(session.batch)})\n\n"
            f"Model: `{item.model_pred}` ({item.model_conf:.2f}){stored}\n\n"
            f"Current: `{session.current_label(position)}`"
            + (" **(changed)**" if session.changed(position) else "")
            + f" · sampled as `{item.sampled_as}`"
        )

    def view(position: int | None):
        valid = position is not None and 0 <= position < len(session.batch)
        current = session.current_label(position) if valid else None
        buttons = [
            gr.update(variant="primary" if name == current else "secondary") for name in label_names
        ]
        return (
            gr.update(value=_contact_sheet(session), selected_index=position if valid else None),
            session.batch[position].path_abs if valid else None,
            detail_text(position),
            session.status_markdown(),
            session.unchanged_hint(),
            gr.update(value=session.mode),
            *buttons,
        )

    with gr.Blocks(title="Photo vs Drawing review", analytics_enabled=False) as demo:
        gr.Markdown("## Photo vs Drawing — review")
        status = gr.Markdown(session.status_markdown())
        with gr.Row():
            mode_radio = gr.Radio(list(MODES), value=session.mode, label="Mode", scale=2)
            quick_toggle = gr.Checkbox(label="Quick mode (click cycles guess → others → skip)", value=False, scale=1)

        with gr.Row():
            with gr.Column(scale=3):
                gallery = gr.Gallery(
                    value=_contact_sheet(session),
                    columns=6,
                    height="auto",
                    object_fit="contain",
                    allow_preview=False,
                    label="Batch (click to select; ✱ = changed)",
                )
            with gr.Column(scale=2):
                preview = gr.Image(
                    label="Preview (full resolution)",
                    type="filepath",
                    height=560,
                    buttons=["fullscreen"],
                )
                detail = gr.Markdown(detail_text(None))
                label_buttons = []
                with gr.Row():
                    for i, name in enumerate(label_names):
                        is_skip = name == session.config.skip_label
                        key = "0/s" if is_skip else str(i + 1)
                        label_buttons.append(
                            gr.Button(
                                f"{name} [{key}]",
                                variant="secondary",
                                elem_id="pvd-skip" if is_skip else f"pvd-label-{i}",
                            )
                        )
                with gr.Row():
                    prev_btn = gr.Button("◀ prev [←]", size="sm", elem_id="pvd-prev")
                    next_btn = gr.Button("next [→] ▶", size="sm", elem_id="pvd-next")

        hint = gr.Markdown(session.unchanged_hint())
        with gr.Row():
            confirm_btn = gr.Button("Confirm & retrain [Enter]", variant="primary", elem_id="pvd-confirm")
            undo_btn = gr.Button("Undo last batch")

        outputs = [gallery, preview, detail, status, hint, mode_radio, *label_buttons]

        def on_select(evt: gr.SelectData):
            raw = evt.index
            position = int(raw[0]) if isinstance(raw, (list, tuple)) else int(raw)
            session.select(position)
            if view_state["quick"]:
                session.cycle_label(position)
            return view(position)

        gallery.select(on_select, inputs=None, outputs=outputs)

        def make_set_label(label: str):
            def set_label_fn():
                if session.selected is not None:
                    session.set_label(session.selected, label)
                return view(session.selected)

            return set_label_fn

        # One button per class (+ skip) so the UI works for any N (G20).
        for name, button in zip(label_names, label_buttons):
            button.click(make_set_label(name), inputs=None, outputs=outputs)

        prev_btn.click(lambda: view(session.move_selection(-1)), inputs=None, outputs=outputs)
        next_btn.click(lambda: view(session.move_selection(+1)), inputs=None, outputs=outputs)

        def on_quick(flag: bool):
            view_state["quick"] = bool(flag)

        quick_toggle.change(on_quick, inputs=[quick_toggle], outputs=None)

        def on_mode(mode: str):
            if mode != session.mode:
                session.set_mode(mode)
                session.last_action = f"Switched to {mode} mode."
            return view(None)

        mode_radio.input(on_mode, inputs=[mode_radio], outputs=outputs)

        def on_confirm():
            session.confirm()
            return view(None)

        def on_undo():
            session.undo()
            return view(None)

        confirm_btn.click(on_confirm, inputs=None, outputs=outputs)
        undo_btn.click(on_undo, inputs=None, outputs=outputs)

    return demo


def run_review(config: Config) -> None:
    """Build the session and launch the app on localhost only (called from the CLI)."""
    session = ReviewSession(config)
    demo = build_app(session)
    # Previews are served straight from the data folders; allow exactly those.
    allowed = [str(p) for p in (config.pool_dir, config.seed_dir, config.test_dir, config.thumbs_dir)]
    demo.launch(server_name="127.0.0.1", share=False, head=KEYBOARD_JS, allowed_paths=allowed)
