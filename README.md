# Photo vs Drawing classifier

A local tool that tells **real photographs** apart from **drawings/artwork** —
including the hard cases: anime and manga art, semi-realistic and hyperrealistic
drawings, and heavily edited or filtered photos (which should still count as
`photo`).

It starts from existing pretrained models (no training from scratch) and improves
through a **human-in-the-loop correction loop**: the model shows its predictions,
you mark what is wrong, the model retrains, repeat.

## How it works

```text
data/pool/   data/seed/<class>/   data/test/<class>/[<tag>/]
      \_____________ | _____________/
                    v
   scan      load, validate, SHA-1, dedupe   -> cache/manifest.parquet
                    v
   embed     CLIP / SigLIP / DINOv2 (+ tiles) -> cache/embeddings/<key>.npz
                    v
   model     CV-tuned logistic probe on cached embeddings
             (zero-shot text prompts until enough labels exist)
             -> models/probe_<timestamp>.joblib
              /                              \
   review (Gradio)                         predict / sort CLI
      -> labels/events.jsonl
                  |
      retrain / audit / eval / finetune
```

The core idea: **embeddings are computed once**; training, sampling and
evaluating operate on the cached matrix, so retraining after a review batch takes
seconds.

- **Zero-shot baseline:** several text prompts per class are averaged into a
  centroid and compared with the image embedding.
- **Linear probe:** a logistic regression on frozen embeddings, with the
  regularization strength chosen by cross-validation.
- **Active learning:** each review batch mixes the most uncertain images with a
  random slice. The random slice gives an unbiased error estimate; the uncertain
  slice is deliberately hard.
- **Honest metrics:** all reported accuracy comes from `data/test/`, which is
  never trained on.
- **Audit tools:** out-of-fold label-error suspects, near-duplicate leakage, and
  shortcut checks (file format, transparency, resolution).
- **Feature blocks:** each image vector is `global` (whole-image embedding),
  optionally `tiles` (native-resolution crops, for texture) and `extra` (the
  deepghs P(real) score). Zero-shot and similarity checks use only `global`.
  A trained model remembers its exact feature recipe, so changing
  `config.yaml` later cannot silently feed it different features.
- **Optional fine-tune:** a `timm` model trained on the collected labels.

Read `LABEL_POLICY.md` before labeling anything — consistency matters more than
any model choice. For a full walkthrough and the theory behind each step, see
[`TUTORIAL.md`](TUTORIAL.md).

## Requirements

- Python **3.11 or 3.12** (PyTorch wheels).
- A GPU is optional: CUDA, Apple MPS and CPU are auto-detected.
- No network needed after the first model download.

## Install

```bash
python3.12 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
```

Then install PyTorch for your machine **first**, then the project:

```bash
# macOS (Apple Silicon, uses MPS) or any CPU-only machine:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# Windows/Linux with an NVIDIA GPU: copy the exact command for your CUDA
# version from https://pytorch.org/get-started/locally/ , for example:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

pip install -e ".[dev]"              # or: pip install -r requirements.txt
# optional extras: deepghs specialist model, iPhone HEIC photos
pip install -e ".[deepghs,heic]"
```

`requirements-lock.txt` lists the exact versions the test suite last passed
with, if you need to reproduce a known-good environment.

## Quickstart

```bash
# 1. Drop your images in:  data/pool/            (unlabeled)
#                          data/seed/photo/      (examples)
#                          data/seed/drawing/
#                          data/test/photo/[tag] (held-out, never trained on)
#                          data/test/drawing/[tag]

pvd scan                 # validate + hash + dedupe, writes cache/manifest.parquet
                         # (re-scans only re-read new/changed files; --full forces all)
pvd embed                # compute and cache embeddings
pvd train                # cross-validated probe -> models/probe_*.joblib
pvd eval                 # honest report on the test set (reports/*.md)

pvd review               # open the active-learning app, correct batches,
                         # confirm (retrains in seconds), repeat

pvd predict --src inbox/ --dst sorted/      # sort new images (adds unsure/ + broken/)
pvd audit                # find likely label mistakes and shortcut risks
pvd ablate               # compare backbones / tiles / extra features
pvd finetune             # optional: go past the linear-probe ceiling
pvd export-labels --format folders --out export/
```

## Commands

| Command | Purpose |
|---|---|
| `pvd scan [--full]` | Build/refresh the manifest; report broken files, duplicates, conflicts, bias warnings |
| `pvd embed [--backbone ID]` | Fill the embedding cache |
| `pvd zeroshot` | Zero-shot prompt-ensemble baseline + test report |
| `pvd baseline-deepghs` | Optional deepghs anime/real specialist on the test set |
| `pvd train` | CV-tuned probe; saves a bundle and a training report |
| `pvd eval [--model PATH]` | Full test-set report for a saved model |
| `pvd review` | Launch the Gradio review app |
| `pvd audit` | Label-error suspects, near-duplicate leakage, shortcut analysis |
| `pvd ablate` | Feature/backbone ablation with a generated recommendation |
| `pvd predict --src --dst [--mode] [--threshold] [--yes]` | Sort a folder (`copy`/`move`/`csv_only`) |
| `pvd finetune` | Fine-tune a timm model on the collected labels |
| `pvd export-labels [--format csv\|folders]` | Export current labels |

Global options: `--config PATH` (default `config.yaml`) and `--verbose` (also
shows full tracebacks). A relative `project_root` in the config is relative to
the config file, so `pvd -c path/to/config.yaml ...` works from any folder.

## Review app

The app runs on `http://127.0.0.1:7860` (this machine only).

- Click a thumbnail to **select** it (does not change the label); the right
  panel shows it at full resolution (use the fullscreen button to zoom).
- Set the label with one button per class, or `skip`. The active label's
  button is highlighted; captions read `label 0.83` (model confidence) or
  `label ✱` once you changed it.
- **Keyboard:** `1`..`9` = class, `0` or `s` = skip, `←`/`→` = previous/next
  image, `Enter` = confirm the batch.
- `Quick mode` makes clicking cycle `model guess -> other classes -> skip`.
- **Modes:** `active` (uncertain + random images), `random` (only random —
  an unbiased spot check), `audit` (likely label mistakes from out-of-fold
  predictions).
- `Confirm & retrain`:
  - in active/random mode, **untouched images are saved as the model's guess**
    ("the model was right") — check every image before confirming;
  - in audit mode, untouched images **keep their current label**; only your
    corrections are saved (as `audit_fix`).
- Retraining after a batch refits the probe with the last chosen `C` (fast);
  a full cross-validated search runs every `review.cv_every` batches.
- Until every class has 5 labels the app uses zero-shot text prompts (or no
  model, for DINOv2) and says so in the status bar.
- `Undo last batch` reverts the last batch (up to `review.undo_depth` in a row);
  those images become candidates again.
- The status bar shows test balanced accuracy (the real progress meter), labels
  per class, and both error rates (whole batch vs the unbiased random slice).

Everything is saved append-only in `labels/events.jsonl`; closing the app never
loses labels.

## Configuration

See `config.yaml` (every field has a default). Key options:

- `classes`: class names in order; works for N ≥ 2 classes. `skip` is reserved.
- `backbone.id`: `clip_b32` (default), `clip_l14`, `siglip_so400m` (224 px),
  `siglip_so400m_384` (384 px, more texture detail), `dinov2_s`, `dinov2_b`
  (no zero-shot), or `fake`/`fake_text` (tests only).
- `backbone.tiles`: native-resolution tiling for texture detail (slower).
- `backbone.extra_features.deepghs_real`: append the deepghs P(real) score
  (needs the `deepghs` extra).
- `review.cv_every`, `review.undo_depth`, `review.retrain_every_batch`: review
  loop behaviour.
- `device`: `auto` (cuda → mps → cpu), or force `cuda`/`mps`/`cpu`.
- `num_workers`: dataloader workers; `0` is always safe (Windows/macOS spawn).

## Troubleshooting

- **`pvd eval`/`pvd train` says there is no test set** — add images under
  `data/test/<class>/` (optionally a tag subfolder like `realistic_drawing`).
  Honest metrics require held-out data.
- **Model/backbone mismatch** — a bundle stores its full feature recipe
  (backbone, tiles, extra block, alpha background); `predict`/`eval` rebuild
  exactly those features. Old bundles without a recipe must match the current
  config; retrain if they don't.
- **MPS op fails** — the embedder automatically retries that backbone on CPU
  and logs it. To let PyTorch fall back per op instead, set
  `PYTORCH_ENABLE_MPS_FALLBACK=1` in your environment (not set by the tool), or
  set `device: cpu` in `config.yaml`.
- **An image is marked `broken` after `pvd embed`** — it passed the header
  check but could not be decoded; it is excluded everywhere. `pvd embed`
  retries previous failures each run.
- **`pvd scan` warns about `bad_label`** — images must sit inside a class
  folder (`data/seed/<class>/`, `data/test/<class>/`), and the folder name must
  be one of `classes`.
- **`pvd baseline-deepghs` says it is not installed** — install the optional
  extra: `pip install dghs-imgutils` (or `pip install -e ".[deepghs]"`).
- **Security note** — model bundles (`models/*.joblib`) are Python pickles:
  loading one runs code. Only use bundles you created or trust. Image decoding
  is restricted to JPEG/PNG/WEBP/BMP/GIF/TIFF (+HEIC with the extra).
- **Many `skip`s per batch** — the pool probably contains out-of-domain images
  (memes, UI screenshots, collages). Clean the pool.
- **Photos of drawings end up as `photo`** — this is the known texture/policy
  conflict (see `LABEL_POLICY.md`); include photographed drawings in the seed and
  test sets and watch the per-tag metrics.

## Project layout

```
src/pvd/
  cli.py          config.py       device.py
  imageio.py      scan.py         features.py
  zeroshot.py     evaluate.py     labels.py
  probe.py        sampler.py      review_app.py
  audit.py        ablation.py     predict.py       finetune.py
  models.py       (load probe/fine-tune bundles, align class order)
  backbones/      base.py clip_backbone.py dinov2_backbone.py deepghs.py fake_backbone.py
tests/            hermetic tests (fake backbone; no downloads)
cache/            manifest.parquet, embeddings/, thumbs/
models/           probe_*.joblib, finetune_*.joblib + .pt
reports/          *.md
labels/           events.jsonl (append-only)
```

## Licensing

This project does not redistribute images or model weights. It downloads weights
at runtime; review their licenses before use:

- OpenCLIP LAION weights (CLIP, SigLIP)
- DINOv2 (`timm`)
- deepghs `anime_real_cls` (OpenRAIL, optional)
- `timm` ImageNet weights

Datasets such as Danbooru-derived sets have their own terms.

## Tests

```bash
pip install -e ".[dev]"
pytest                 # hermetic suite, no downloads
pytest -m slow         # opt-in: real clip_b32 (downloads ~600 MB once)
```

Tests never download models: they use the deterministic `fake` / `fake_text`
backbones and generate synthetic images at runtime. `tests/test_regressions.py`
guards every bug fixed after the first code review.
