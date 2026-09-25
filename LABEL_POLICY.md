# Label policy

The model can only be as consistent as its labels. This file is the contract for
how images are labeled. The defaults below are configurable in `config.yaml`
(especially `classes`), but changing them halfway through a review session will
undo progress, so decide early.

## Guiding question

> **"Is this image a camera capture of a real scene?"**

- **Yes -> `photo`**. Editing, filters, black & white, and cropping do not change
  the origin.
- **No, it is hand-made or digitally painted 2D art -> `drawing`**.

Label by **how the depicted content was made**, not by the file's history.

## Decisions confirmed by Dev (2026-09-24)

The plan left three cases open until they were confirmed. They are now settled
and match the table below:

| Open question | Decision |
|---|---|
| AI-generated realistic images | `skip` (no `ai_generated` class for now) |
| 3D renders, CGI, 3D game screenshots | `skip` (no `cg_render` class for now) |
| Photo of a painting in a museum (frame, wall, people visible) | `photo` |

Changing any of these later means relabeling the affected images, so record the
date and the reason here if you do.

## Edge cases

| Case | Label | Rationale |
|------|-------|-----------|
| Heavily filtered / color-graded / B&W photo | `photo` | Editing does not change origin |
| Cosplay, figurines, statues, dolls, props | `photo` | Camera capture of a real object |
| Photo or scan of a drawing/painting on paper/canvas | `drawing` | Content is art (see the texture caveat below) |
| Photo of a painting in a museum (frame, wall, people visible) | `photo` | The scene is real; the art is a small part |
| Screenshot of anime/game art | `drawing` | |
| 3D renders, CGI, game screenshots (3D) | `skip` | Neither photo nor 2D drawing |
| AI-generated realistic images | `skip` | Detecting AI generation is a separate problem |
| AI-generated anime/illustration | `drawing` | Art-styled content |
| Photo with drawn overlays / anime character composited on a real background | `skip` | Mixed; unclear majority |
| Manga pages, comics, sketches, line art | `drawing` | |
| Pixel art | `drawing` | |
| Memes, text-heavy images, UI screenshots, charts | `skip` | Out of domain |
| Collages mixing both | `skip` | |

`skip` is a **reserved pseudo-label**, never a class. Skipped images are excluded
from training and will not appear in future review batches.

## The texture caveat (important)

A phone photo of a pencil drawing contains real camera noise and paper grain.
Texture-based features will push it toward `photo`, while this policy says
`drawing`. To keep the model honest:

- include photographed/scanned drawings in both the seed and test sets,
- watch the per-tag metrics (e.g. tag `photographed_drawing`),
- treat this conflict as a known limitation, not a bug.

## AI-generated images

Detecting AI generation is a different problem from photo-vs-drawing: the evidence
is generator-specific artifacts and detectors decay as new generators appear.

1. Default: label AI-generated realistic images `skip`.
2. To track them, add a class `ai_generated` to `classes` and accept moderate
   accuracy, tracked separately in per-tag metrics. Also add zero-shot prompts
   for it under `zeroshot.prompts`, or the review app stays in "labels needed"
   mode until every class has 5 labels.
3. A dedicated detector can be added later as a separate stage.
