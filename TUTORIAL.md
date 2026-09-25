# Tutorial: using `pvd` and understanding how it works

This guide has two halves:

- **Part 1 — Using it.** A full walkthrough from an empty folder to a model that
  sorts your images, with what each command produces and what to look at.
- **Part 2 — How it works.** The theory behind each step, in plain words but
  without skipping the technical parts.

The [README](README.md) is the short reference; this file is the long version.
If you only want to label and sort images, Part 1 is enough. If you want to know
*why* a number looks the way it does, read Part 2.

---

## Contents

**Part 1 — Using it**

1. [The mental model in one minute](#1-the-mental-model-in-one-minute)
2. [Install](#2-install)
3. [Prepare your data](#3-prepare-your-data)
4. [Scan: check your data](#4-scan-check-your-data)
5. [Embed: turn images into vectors](#5-embed-turn-images-into-vectors)
6. [Baselines: zero-shot (and deepghs)](#6-baselines-zero-shot-and-deepghs)
7. [Train and evaluate a probe](#7-train-and-evaluate-a-probe)
8. [Review: the correction loop](#8-review-the-correction-loop)
9. [Audit: am I fooling myself?](#9-audit-am-i-fooling-myself)
10. [Predict: sort new images](#10-predict-sort-new-images)
11. [Going further: ablation, tiles, fine-tuning](#11-going-further-ablation-tiles-fine-tuning)
12. [Everyday recipes](#12-everyday-recipes)

**Part 2 — How it works**

13. [Embeddings: images as points in space](#13-embeddings-images-as-points-in-space)
14. [Zero-shot classification with text prompts](#14-zero-shot-classification-with-text-prompts)
15. [The linear probe](#15-the-linear-probe)
16. [Choosing C with cross-validation](#16-choosing-c-with-cross-validation)
17. [Active learning: which images to show you](#17-active-learning-which-images-to-show-you)
18. [Labels as an event log](#18-labels-as-an-event-log)
19. [Honest evaluation](#19-honest-evaluation)
20. [Audit techniques](#20-audit-techniques)
21. [Texture, tiles and the 224-pixel problem](#21-texture-tiles-and-the-224-pixel-problem)
22. [Fine-tuning](#22-fine-tuning)
23. [Caching and reproducibility](#23-caching-and-reproducibility)
24. [Glossary](#24-glossary)

---

# Part 1 — Using it

## 1. The mental model in one minute

```text
your images ──► scan ──► embed ──► a vector per image (cached once)
                                        │
            ┌───────────────────────────┼─────────────────────────┐
            ▼                           ▼                         ▼
   zero-shot (text prompts)     linear probe (learned)     audit / eval
            │                           │
            └────────► review app ◄─────┘   you correct mistakes,
                          │                 the probe retrains in < 1 s
                          ▼
                 labels/events.jsonl ──► train ──► predict (sort folders)
```

Three ideas carry the whole project:

1. **A big pretrained model (CLIP) already "understands" images.** We never
   train it. We only use it to turn each image into a list of numbers (an
   *embedding*) once, and cache the result.
2. **A tiny model on top does the actual photo-vs-drawing decision.** It is a
   logistic regression, so it trains in well under a second on thousands of
   images. That is what makes the correction loop feel instant.
3. **You are part of the loop.** The app shows you the images the model is
   least sure about; you fix its mistakes; it retrains; repeat. Accuracy is
   always measured on a separate test set that is never trained on.

## 2. Install

You need Python 3.11 or 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

Install PyTorch for your machine first, then the project:

```bash
# Mac (Apple Silicon) or CPU-only:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# NVIDIA GPU: copy the command for your CUDA version from pytorch.org, e.g.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

pip install -e ".[dev]"
```

Optional extras:

```bash
pip install -e ".[heic]"      # read iPhone .heic photos
pip install -e ".[deepghs]"   # the deepghs anime/real specialist model
```

Check it works:

```bash
pvd --help
pytest            # ~140 tests, no downloads, ~20 s
```

The first command that needs CLIP (`pvd embed`) downloads its weights once
(about 600 MB for `clip_b32`). After that nothing needs the network.

## 3. Prepare your data

Everything lives under `data/`, in three roles:

```text
data/
├── pool/                    unlabeled images you want to review
│   └── (any subfolders)
├── seed/                    examples you already know the answer for
│   ├── photo/
│   └── drawing/
└── test/                    held-out images, NEVER trained on
    ├── photo/
    │   ├── filtered_photo/      ← optional "tag" subfolders
    │   └── cosplay/
    └── drawing/
        ├── anime/
        └── realistic_drawing/
```

**What each folder is for**

| Folder | Labeled? | Used for | How many |
|---|---|---|---|
| `seed/<class>/` | yes, by folder | training from day one | 50–200 per class is a good start |
| `test/<class>/[<tag>/]` | yes, by folder | measuring accuracy only | 30+ per class; more for each hard tag |
| `pool/` | no | what the review app shows you | as many as you like |

**Rules that save you pain later**

- **Read [`LABEL_POLICY.md`](LABEL_POLICY.md) first.** The model can only be as
  consistent as your labels. The guiding question is *"Is this a camera capture
  of a real scene?"* A filtered photo is still `photo`; a photo *of* a drawing
  on paper is `drawing`; memes, UI screenshots, collages, CG renders and
  AI-realistic images are `skip`.
- **Make the test set hard on purpose.** Put the cases you care about in tag
  folders: `realistic_drawing`, `filtered_photo`, `anime`, `photographed_drawing`,
  `cosplay`. You'll get a separate accuracy for each tag, which is how you find
  out *where* the model fails, not just how often.
- **Don't let the test set look like the training set.** Near-copies (the same
  photo resized, the same artwork from two sites) make accuracy look better than
  it is. The audit command finds them.
- **Watch out for shortcuts.** If every drawing you collected is a transparent
  PNG and every photo is a JPEG, the model can learn "PNG = drawing" instead of
  anything about the picture. Mix formats in both classes where you can. The
  scan report warns you about this.
- Files must sit **inside** a class folder. A file directly in `data/test/` is
  flagged `bad_label` and ignored.
- Supported formats: JPEG, PNG, WEBP, BMP, GIF (first frame), TIFF, and HEIC
  with the extra installed.

## 4. Scan: check your data

```bash
pvd scan
```

Scan walks the three folders and, for every file:

- computes its **SHA-1 hash** (a fingerprint of the bytes),
- opens and fully decodes it to catch broken files,
- records size, format, colour mode, transparency, frame count,
- detects **duplicates** (same hash) and **label conflicts** (the same file in
  both `seed/photo` and `seed/drawing`).

It writes `cache/manifest.parquet` (the table everything else reads) and a
report `reports/<time>_scan.md`. **Open the report.** Look at:

- **Counts by split/class** — are the classes roughly balanced?
- **Excluded files** — each with a reason: `broken`, `too_large`, `duplicate`,
  `label_conflict`, `bad_label`.
- **Early bias warnings** — the file-format and transparency share per class.
  A line like `seed/drawing: PNG=92%` next to `seed/photo: PNG=3%` is a shortcut
  waiting to happen.

Re-running `pvd scan` is cheap: unchanged files (same size and modification
time) are reused from the previous scan. `pvd scan --full` re-reads everything.

Duplicates are resolved by priority **test > seed > pool**: if an image is in
both `test` and `pool`, the test copy wins and the pool copy is ignored, so the
test set stays clean.

## 5. Embed: turn images into vectors

```bash
pvd embed
```

Each usable image goes through the backbone (by default CLIP ViT-B/32) and
becomes a 512-number vector, saved in `cache/embeddings/clip_b32.npz`, keyed by
the image's SHA-1. This is the slow step (seconds per hundred images on a GPU,
longer on CPU) and it happens **once per image**:

- run it again → 0 new embeddings;
- rename or move a file → 0 new embeddings (the hash didn't change);
- interrupt it (Ctrl-C) → it resumes where it stopped.

Images that pass the scan but can't be decoded here are marked `broken` in the
manifest and skipped everywhere else.

## 6. Baselines: zero-shot (and deepghs)

Before training anything, see how far the pretrained model gets on its own:

```bash
pvd zeroshot
```

This compares each test image with text descriptions ("a photograph",
"a smartphone photo", … vs "a drawing", "an anime illustration", …) and picks
the closer class. It needs **no labels at all**. The report
(`reports/<time>_zeroshot.md`) is your baseline: anything you train later
should beat it, especially on the hard tags.

If you installed the deepghs extra, also run the specialist anime/real model:

```bash
pvd baseline-deepghs
```

It was trained on a lot of anime-vs-real data, so compare its `anime` tag score
with its `realistic_drawing` score: a specialist is often strong on what it was
trained for and weak just outside it.

## 7. Train and evaluate a probe

```bash
pvd train
pvd eval
```

`train` fits a logistic regression on the embeddings of every labeled image
(seed folders + anything you labeled in the review app), picking its
regularization strength by cross-validation. It needs **at least 5 labeled
images per class**. It saves `models/probe_<time>.joblib` and a report.

`eval` scores the newest model on the test set. The report has:

| Section | What to look at |
|---|---|
| Accuracy / balanced accuracy / ROC-AUC | Balanced accuracy is the headline number (it isn't fooled by class imbalance). |
| Per-class precision/recall | Is one class systematically missed? |
| Confusion matrix | Rows = truth, columns = prediction. Off-diagonal = mistakes. |
| **Per-tag** | The most useful table: accuracy for `realistic_drawing`, `filtered_photo`, … |
| Bias breakdowns | Accuracy by format, transparency, resolution, source folder. A big gap means a shortcut. |
| Coverage vs accuracy | How many images you can sort automatically at each confidence threshold, and how accurate those are. Gives a **recommended `unsure_threshold`**. |
| Worst test errors | The 12 most confident mistakes. Open them — they're either label errors or the model's blind spots. |

## 8. Review: the correction loop

```bash
pvd review
```

This opens a web page at `http://127.0.0.1:7860` (only reachable from your
computer).

```text
┌──────────────────────────────────────────────────────────────────┐
│ Mode: active · training labels · Model: … · Test balanced acc    │
│ ( ) active  ( ) random  ( ) audit          [ ] Quick mode        │
├───────────────────────────────┬──────────────────────────────────┤
│ 24 thumbnails                 │ full-resolution preview          │
│ "drawing 0.62"  "photo ✱" …   │ file name · model guess · current│
│                               │ [photo 1] [drawing 2] [skip 0/s] │
├───────────────────────────────┴──────────────────────────────────┤
│ ⚠ Untouched images are saved as the model's guess                │
│ [Confirm & retrain]  [Undo last batch]                           │
└──────────────────────────────────────────────────────────────────┘
```

**One batch, step by step**

1. Look over the thumbnails. Each caption is the current label and the model's
   confidence (e.g. `drawing 0.62`). Low confidence = the model is unsure.
2. Click a thumbnail to see it at full resolution on the right (use the
   fullscreen button to zoom; realistic drawings often only give themselves
   away up close). Clicking **never** changes a label.
3. If the label is wrong, press a label button (or a key). The caption turns
   into `photo ✱` — the ✱ marks your changes.
4. When every thumbnail shows the right label, press **Confirm** (or `Enter`).

**Keyboard**

| Key | Action |
|---|---|
| `1`, `2`, … | set the selected image to class 1, 2, … |
| `0` or `s` | skip (out of scope; never trained on, never shown again) |
| `←` / `→` | previous / next image |
| `Enter` | confirm the batch |

**The one rule to remember:** in active and random mode, **every image you
didn't touch is saved as the model's guess**. That's what makes the loop fast —
you only fix mistakes — but it also means confirming without looking stores
the model's errors as truth. If in doubt, `skip`.

**What happens on Confirm**

1. Your labels are appended to `labels/events.jsonl` (safe even if the app
   crashes a second later).
2. The probe is refit with the new labels (fraction of a second). Every 10
   batches it also re-runs the full cross-validation.
3. Every pool image is re-scored, the test set is re-evaluated, and the next
   batch is chosen.

**Reading the status bar**

- `Model:` tells you what's making the predictions. Until every class has 5
  labels it's *zero-shot* (text prompts), or *none* if your backbone can't do
  zero-shot (DINOv2).
- `Test balanced accuracy` is the real progress meter, with the change since the
  last batch in percentage points.
- `Last batch: changed 5/24 (random part: 1/6 wrong)` — the whole-batch number
  stays high even when the model improves, because the app *deliberately* shows
  you hard images. The random part is the honest estimate. Section 17 explains
  why.

**Modes**

- **active** (default): mostly uncertain images plus a random 25%.
- **random**: only random images. Use it now and then as a spot check of how
  the model does on "ordinary" pool images.
- **audit**: images you (or the seed folders) labeled that the model strongly
  disagrees with. In audit mode untouched images **keep their current label**;
  only your corrections are saved.

**Undo** reverts the last confirmed batch (up to 20 in a row); those images go
back to the pool. Nothing is ever deleted from the log — undo is just another
event.

You can close the app any time; restarting it picks up exactly where you were.
When the pool is empty it tells you so — run `pvd train` and `pvd eval` for a
final report.

## 9. Audit: am I fooling myself?

```bash
pvd audit
```

The report (`reports/<time>_audit.md`) has three checks:

1. **Label-error suspects.** Labeled images whose label the model finds very
   unlikely (under 20%) — computed so the model has *not* seen the image during
   training (section 20). Often these are genuine labeling mistakes, sometimes
   genuinely ambiguous images; review them in the app's audit mode.
2. **Near-duplicate leakage.** Test images that are almost identical (cosine
   similarity > 0.97) to a training image. Each one inflates your test score;
   move or delete one of the pair.
3. **Shortcut analysis.** Can the class be guessed from the file format,
   transparency or resolution alone? If yes (and clearly better than always
   guessing the bigger class), the model may be learning the shortcut. Fix it by
   adding counter-examples (drawings saved as JPEG, photos as PNG, …).

## 10. Predict: sort new images

```bash
pvd predict --src ~/Downloads/new_images --dst ~/Pictures/sorted
```

Result:

```text
sorted/
├── photo/
├── drawing/
├── unsure/          confidence below the threshold
├── broken/          files that couldn't be read
└── predictions.csv  one row per input file
```

- `--threshold 0.9` — images below this confidence go to `unsure/`. Pick it
  from the coverage table in the eval report, not by feel.
- `--mode copy` (default), `--mode csv_only` (just the CSV), or
  `--mode move --yes` (moves originals; `--yes` is required on purpose).
- `--model models/probe_….joblib` — use a specific model (default: newest).
- Files are never overwritten; name clashes get `_<hash>` appended.
- The CSV is written *before* any file is moved, so an interrupted move still
  leaves a record. Columns: `path, sha1, pred, confidence, p_photo, p_drawing,
  bucket, dest, error`.

## 11. Going further: ablation, tiles, fine-tuning

**Which backbone is best for my data?**

```bash
pvd ablate
```

Trains the same probe on several feature setups (CLIP B/32, CLIP L/14,
SigLIP, DINOv2, with/without tiles, with/without deepghs) and writes a table
with cross-validated accuracy, test accuracy, accuracy on `realistic_drawing`
and `filtered_photo`, and embedding speed, plus a recommendation computed from
those numbers. It downloads each model the first time, and embeds everything
once per setup, so run it when you have time.

To switch, set `backbone.id` in `config.yaml` and run `pvd embed` then
`pvd train`. Options: `clip_b32` (fast default), `clip_l14`, `siglip_so400m`,
`siglip_so400m_384` (higher resolution), `dinov2_s`, `dinov2_b` (no zero-shot).

**Tiles** (`backbone.tiles.enabled: true`) add texture evidence from
full-resolution crops. Slower to embed; often helps with hyperrealistic
drawings. See section 21.

**Fine-tuning**

```bash
pvd finetune
```

Trains a whole ConvNeXt network on your labels. Much slower (GPU strongly
recommended), and only worth it once you have a few thousand labels and the
probe has plateaued. The report compares it with the probe on the same test
set, per tag. `pvd predict` and `pvd eval` accept either model type.

## 12. Everyday recipes

| I want to… | Do this |
|---|---|
| add new images to review | drop them in `data/pool/`, `pvd scan`, `pvd embed`, `pvd review` |
| fix a seed label | move the file to the right `seed/<class>/` folder and re-scan, or fix it in audit mode |
| export my labels | `pvd export-labels --format csv` (or `folders` for one folder per class, as links) |
| start labeling from scratch | delete `labels/events.jsonl` (keeps seeds; loses review labels) |
| add a third class | add it to `classes` in `config.yaml`, create `seed/<class>/` and `test/<class>/`, add zero-shot prompts for it under `zeroshot.prompts` |
| use a different config | `pvd -c other/config.yaml <command>` (paths are relative to that file) |
| see full error details | add `-v`: `pvd -v train` |
| run on another machine | copy the project folder; caches and labels are keyed by file content, not path |

---

# Part 2 — How it works

## 13. Embeddings: images as points in space

An **embedding** is a fixed-length list of numbers that summarizes an image.
CLIP ViT-B/32 produces 512 numbers per image. You can think of each image as a
point in a 512-dimensional space.

**Where the numbers come from.** CLIP is a *vision transformer*: it cuts the
image (resized to 224×224) into 32×32-pixel patches, turns each patch into a
vector, and lets the patches exchange information through many attention layers.
The final summary vector is the embedding.

**Why the space is useful.** CLIP was trained on hundreds of millions of
(image, caption) pairs with a *contrastive* objective: the embedding of an image
must be close to the embedding of its own caption and far from other captions.
To succeed at that, the model had to learn concepts that captions talk about —
"anime", "oil painting", "photo taken at night". So images that are alike in
those ways end up close together, even though nobody taught CLIP our two
classes.

**Distance = cosine similarity.** Every embedding is *L2-normalized* (scaled to
length 1). For unit vectors, the dot product equals the cosine of the angle
between them:

```text
cos(a, b) = a · b = Σ aᵢ bᵢ        (1 = same direction, 0 = unrelated)
```

That single operation is used everywhere: zero-shot, the diversity filter, and
near-duplicate detection.

**Why we freeze the model.** Running CLIP is the expensive part. Doing it once
and caching the vectors means everything else — training, sampling, evaluating —
is matrix arithmetic on a table of numbers, which takes milliseconds.

**Backbones** differ in what they learned:

| id | Model | Vector size | Input | Notes |
|---|---|---|---|---|
| `clip_b32` | CLIP ViT-B/32 | 512 | 224 px | fast default |
| `clip_l14` | CLIP ViT-L/14 | 768 | 224 px | smaller patches, more detail, slower |
| `siglip_so400m` / `_384` | SigLIP | 1152 | 224 / 384 px | trained with a sigmoid loss; 384 keeps more texture |
| `dinov2_s` / `dinov2_b` | DINOv2 | 384 / 768 | 518 px | self-supervised, no text: strong on texture, no zero-shot |

## 14. Zero-shot classification with text prompts

Because CLIP embeds text into the *same* space as images, we can classify
without any labeled image:

1. For each class, embed several prompts (a **prompt ensemble**):
   `photo` → "a photograph", "a smartphone photo", "a DSLR photograph", …
2. Average them and re-normalize → one **centroid** per class. Averaging cancels
   out the quirks of any single phrasing.
3. For an image embedding `x`, compute its similarity to each centroid `cₖ`,
   multiply by CLIP's learned temperature (≈100), and apply **softmax**:

```text
p(class k | x) = exp(100 · x·cₖ) / Σⱼ exp(100 · x·cⱼ)
```

**Limits.** These "probabilities" aren't calibrated: 0.9 does not mean "right
90% of the time". They're good for ranking, not for thresholds. And zero-shot
only knows what the prompts say, so it struggles exactly where words are
ambiguous (a hyperrealistic painting *is* "a realistic image"). That's why the
tool moves to a learned probe once you have labels.

## 15. The linear probe

The probe is a **logistic regression** on the frozen embeddings. For two classes
it learns a weight vector `w` (512 numbers) and a bias `b`:

```text
p(drawing | x) = σ(w · x + b),     σ(z) = 1 / (1 + e^(−z))
```

Geometrically, `w · x + b = 0` is a flat boundary (a *hyperplane*) cutting the
512-dimensional space in two; `σ` turns the distance from that boundary into a
probability. With N classes it learns one `w` per class and uses softmax instead
of σ — which is why every part of the tool works for any N ≥ 2.

**Why something this simple works.** CLIP has already done the hard work of
arranging images meaningfully. If photos and drawings sit in different regions
of the space, a single hyperplane separates them well. Training takes a fraction
of a second, so the model can be refit after every review batch.

**Class weights.** With `class_weight: balanced`, each class's examples are
weighted by `n_total / (n_classes · n_class)`. If you have 1000 photos and 100
drawings, each drawing counts 10× more, so the model can't win by always saying
"photo".

**Regularization (`C`).** The training loss is

```text
loss = C · Σ (cross-entropy on each labeled image)  +  ½ ‖w‖²
```

The `½‖w‖²` term pulls the weights toward zero. **Small `C`** → strong pull →
a simple, cautious model with probabilities near 0.5. **Large `C`** → the model
fits the labels closely, with sharper probabilities, but risks memorizing
noise. The next section is how `C` gets chosen.

## 16. Choosing C with cross-validation

We can't choose `C` by testing on the test set — that would leak test
information into the model. Instead, **k-fold cross-validation** is run on the
training labels only:

1. Split the labeled images into 5 folds, keeping the class ratio in each
   (*stratified*).
2. For each candidate `C` in `[0.01, 0.1, 1, 10, 100]`: train on 4 folds,
   score on the 5th, rotate so every fold is scored once, and average.
3. The score is **balanced accuracy** — the average of per-class recall — so a
   majority-class guesser scores 50%, not 90%.
4. Among candidates that tie for the best score, pick the **largest** `C`. A tie
   means the extra regularization bought nothing, and a sharper model gives more
   useful confidence values for the `unsure` threshold.
5. Refit on *all* labels with that `C`.

In the review app the full search runs at startup and every 10 batches; in
between, the probe is refit once with the last chosen `C` (fast path).

## 17. Active learning: which images to show you

Labeling is the expensive resource, so the app tries to show you the images
whose labels teach the model the most.

**Uncertainty sampling.** For every unlabeled pool image, take the two highest
class probabilities and compute the **margin**:

```text
margin = p_top1 − p_top2        (0 = coin flip, 1 = certain)
```

The images with the smallest margin sit right next to the decision boundary.
A label there moves the boundary the most; a label on an image the model already
gets right with 0.99 confidence teaches it almost nothing.

**Diversity filter.** Uncertain images often come in clusters (ten crops of the
same comic). An uncertain image is skipped if its cosine similarity to one
already picked for this batch is above 0.95; the batch is then topped up with
random picks.

**The random slice (25%).** Part of every batch is drawn uniformly at random
from the pool. Two reasons:

1. **Unbiased measurement.** The uncertain images are hard *by construction*, so
   the model's error rate on them stays high even as the model improves. The
   random slice is a fair sample of the pool, so its error rate estimates how the
   model does on ordinary images.
2. **Exploration.** Uncertainty sampling can fixate on one confusing region
   (blank images, extreme crops). Random picks keep showing you the rest.

**Shuffling.** The batch is shuffled so you can't tell which thumbnails are the
"hard" ones — knowing that would bias your judgment.

## 18. Labels as an event log

Labels are stored as an **append-only log** (`labels/events.jsonl`), one JSON
object per line:

```json
{"ts": "2026-09-24T21:14:03Z", "sha1": "ab12…", "label": "drawing",
 "source": "review", "batch_id": "b0007", "model_pred": "photo",
 "model_conf": 0.61, "sampled_as": "uncertain"}
```

The current label of an image is computed by **replaying** the log: the last
event for that SHA-1 wins, ignoring batches that a later `undo` event reverted.
This pattern is called *event sourcing*. It gives you:

- **crash safety** — a write is one line, flushed to disk immediately; a torn
  last line after a crash is skipped and the next write starts on a new line;
- **undo** — just another event, so nothing is ever destroyed;
- **history** — you can later ask "how often was the model wrong on uncertain
  vs random picks?" because `model_pred` and `sampled_as` are recorded.

**Effective training labels** = seed folder labels, overridden by any logged
label for the same image, minus `skip`, minus anything in the test split
(test images can never be trained on, even if a label points at them).

## 19. Honest evaluation

**Held-out test set.** All reported accuracy comes from `data/test/`, which is
never trained on. Training accuracy is meaningless for judging a model: a model
that memorized its training set scores 100% there and can still be useless.

**Balanced accuracy.** The mean of per-class recall. With 90 photos and 10
drawings, "always photo" has 90% accuracy but 50% balanced accuracy.

**ROC-AUC.** The probability that a random drawing gets a higher "drawing" score
than a random photo. It measures ranking quality independent of any threshold.

**Per-tag metrics.** Overall accuracy averages over easy and hard cases. The
per-tag table exposes the failure that matters: "99% overall, 60% on
`realistic_drawing`" is a very different model from "99% everywhere".

**Coverage vs accuracy.** For a threshold `t`, images with max probability < `t`
are left undecided (`unsure`). *Coverage* is the share decided; *accuracy* is
measured on the decided ones. Raising `t` trades coverage for accuracy. The
report recommends the lowest `t` whose decided accuracy reaches 98% — that's the
threshold to give `pvd predict`.

**Don't overfit the test set.** Every time you tweak something because of a test
number, the test set becomes a little less "unseen". If you've iterated a lot,
refresh it with new images.

## 20. Audit techniques

**Out-of-fold predictions (label errors).** To judge a *label*, you need a
prediction from a model that did not see that image. The audit splits the
labeled images into folds, trains on all but one fold, predicts the held-out
fold, and rotates — so every image gets a prediction from a model that never
saw it. An image whose own label gets under 20% probability is a suspect. (If
the model had trained on it, it would simply echo the label back and hide the
mistake.)

**Near-duplicates.** For each test image, the highest cosine similarity to any
labeled training image. Above 0.97 means it's essentially the same picture
(re-encoded, resized, slightly cropped), so the test score for it is not a real
test.

**Shortcut learning.** A model learns whatever separates the classes in the
training data — including things you didn't intend: file format, transparency,
resolution, the website an image came from. The audit asks two questions for
each such attribute:

1. How well does the attribute *alone* predict the class? It warns when this is
   ≥ 95% **and** at least 10 points better than always guessing the majority
   class (so plain class imbalance doesn't trigger it).
2. How well can the attribute be predicted *from the embedding*? High means the
   embedding carries that information, so the probe *could* use it.

Transparent PNGs are a classic case: anime cutouts are often transparent, and a
transparent background rendered as black or white is an easy visual cue. That's
why the loader always composites transparency onto the same neutral background
and records `has_alpha` for this check.

## 21. Texture, tiles and the 224-pixel problem

A hyperrealistic pencil drawing and a black-and-white photo can look identical
at 224×224 pixels. The evidence that separates them is **fine texture**:

| Photos have | Drawings have |
|---|---|
| sensor noise, demosaicing patterns | paper grain, pencil/brush strokes |
| lens blur and depth of field | uniform sharpness, hatching |
| JPEG artifacts from the camera | canvas texture, visible layers |

Downscaling a 4000-pixel image to 224 pixels throws almost all of that away.

**Tiles** keep some of it. With `backbone.tiles.enabled`:

1. Downscale only if the long side exceeds `max_side` (1536 px).
2. Cut `count` crops (default 5: centre + 4 random) of the backbone's input size
   **at that resolution** — each crop shows real pixels, not a thumbnail.
   Crop positions are seeded by the image's hash, so they're identical every run.
3. Embed each crop, average, normalize → the **tiles block**.

The final vector is `[global block | tiles block]` (plus the optional deepghs
score). The global block still describes the whole scene; the tiles block adds
texture. Zero-shot and similarity checks use only the global block, because
the concatenated vector has length √2 and dot products stop being cosines.

**The policy conflict.** A phone photo of a pencil drawing has real camera noise
— texture features push it toward `photo`, while the label policy says
`drawing`. There's no feature trick that resolves this; it's a real ambiguity.
Include photographed drawings in both seed and test sets (with a tag) and watch
that tag's accuracy.

## 22. Fine-tuning

The probe can only draw one flat boundary in a fixed space. **Fine-tuning**
retrains the whole network (a ConvNeXt-Tiny pretrained on ImageNet) so the
space itself adapts to your classes.

Choices, and why:

- **Two learning rates.** The new classifier head starts from random weights and
  needs a large step size (1e-3); the pretrained backbone is already good and
  needs gentle changes (1e-4), or it forgets what it knew.
- **Warmup then cosine decay** of the learning rate: small steps at first while
  the head is random, then a smooth decrease.
- **Label smoothing (0.1)**: targets of 0.95/0.05 instead of 1/0 discourage
  over-confident predictions.
- **Class-weighted loss**, like the probe.
- **Augmentation that preserves the evidence.** Used: random crops, flips, mild
  colour jitter, occasional grayscale (B&W photos and sketches exist in both
  classes), occasional JPEG re-compression. Avoided: strong blur, strong noise,
  heavy downscaling — those erase or fake exactly the texture cues from
  section 21.
- **Early stopping** on validation balanced accuracy (15% of the labels held out).

Fine-tuning needs many more labels than the probe and a GPU; it's the last step,
not the first.

## 23. Caching and reproducibility

**Content addressing.** Caches and labels are keyed by the SHA-1 of each file's
bytes, not by its path. Renaming or moving files changes nothing; editing a file
(new bytes) makes it a new image.

**Feature recipes.** The embedding cache file name encodes everything that
changes the numbers: backbone, tiles (count and max side), the deepghs block,
and the alpha background (e.g. `clip_b32+tiles5`). Two recipes can never share
a cache.

**Model bundles.** A saved model stores its class list, backbone, full feature
recipe, feature size, chosen `C`, cross-validation score, test metrics, the
config, library versions, and a hash of the exact labels it was trained on.
`predict` and `eval` rebuild features with the *model's* recipe, not whatever
`config.yaml` says today, and reorder class columns by name.

**Checkpoints.** Embedding writes small checkpoint files every few batches, so
an interrupted run resumes; they're merged into one file at the end.

## 24. Glossary

| Term | Meaning |
|---|---|
| **Active learning** | Choosing which examples to label so each label helps the model the most. |
| **Backbone** | The big pretrained network that turns images into embeddings (CLIP, SigLIP, DINOv2). |
| **Balanced accuracy** | Mean of per-class recall; not fooled by class imbalance. |
| **Block** | One part of a feature vector: `global`, `tiles` or `extra`. |
| **Calibration** | Whether "0.9 confidence" really means right 90% of the time. |
| **Centroid** | The average (re-normalized) of several vectors, e.g. prompt embeddings. |
| **Cosine similarity** | Dot product of unit vectors; 1 = same direction, 0 = unrelated. |
| **Cross-validation** | Rotating train/validation splits of the training data to tune settings without touching the test set. |
| **Embedding** | A fixed-length vector summarizing an image (or text). |
| **Event sourcing** | Storing every change as an appended event and rebuilding state by replay. |
| **Linear probe** | A linear classifier (here logistic regression) trained on frozen embeddings. |
| **Margin** | Difference between the top two class probabilities; small = uncertain. |
| **Out-of-fold** | A prediction for an example made by a model that didn't train on it. |
| **Pool** | Unlabeled images waiting for review. |
| **Regularization (`C`)** | Penalty on large weights; small `C` = simpler, more cautious model. |
| **Seed set** | Images labeled by folder before any review. |
| **SHA-1** | A 40-character fingerprint of a file's bytes; identical files share it. |
| **Shortcut learning** | A model relying on an unintended cue (file format, background) instead of content. |
| **Softmax** | Turns scores into probabilities that sum to 1. |
| **Tag** | Optional subfolder of the test set used for per-category metrics. |
| **Test set** | Held-out labeled images used only for measuring accuracy. |
| **Zero-shot** | Classifying with text descriptions instead of labeled examples. |
