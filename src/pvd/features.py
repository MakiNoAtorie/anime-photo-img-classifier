"""Embedding computation and content-addressed cache (G2, G24, G26).

Design: embeddings are computed once per (image bytes, feature recipe) pair and
stored in ``cache/embeddings/<feature_key>.npz``. Training, sampling and
evaluation then operate on the cached matrix and are fast (NF2).

The cache key is the **SHA-1 of the file bytes**, so renames and moves never
trigger recomputation. Every setting that changes the numbers (backbone, tile
settings, extra features, alpha background) is part of the ``feature_key``, so
caches for different feature recipes can never be mixed (G8).

A feature vector is a concatenation of *blocks*, each recorded in the cache meta
(``block_sizes``):

- ``global``: the backbone embedding of the whole image, L2-normalized. This is
  the only block that is text-aligned (zero-shot) and the one used for cosine
  similarity (diversity filter, near-duplicate audit).
- ``tiles`` (optional, G24): mean of native-resolution crop embeddings,
  L2-normalized, so texture evidence survives the 224 px downscale.
- ``extra`` (optional, G26): the deepghs P(real) score, mapped to ``2p - 1``.
  The transform is fixed (not fitted on data) so a cached value never goes
  stale when new images are added.

Images that cannot be embedded are **not** silently turned into zero vectors:
they are recorded in the cache meta as failures and reported through a validity
mask so callers can drop them (and ``pvd embed`` marks them ``broken``).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .backbones import build_backbone
from .config import Config, TilesConfig
from .device import pick_device
from .imageio import load_rgb
from .scan import load_manifest, save_manifest

log = logging.getLogger(__name__)

# Write a checkpoint shard this often, so an interrupted run resumes (NF1).
CHECKPOINT_EVERY_BATCHES = 10
CHECKPOINT_EVERY_IMAGES = 200
# Merge shards back into the main file once this many have accumulated.
MAX_SHARDS = 20

DEFAULT_BACKGROUND = (255, 255, 255)


# --------------------------------------------------------------------------- #
# Feature key and recipe
# --------------------------------------------------------------------------- #
def feature_key(backbone_id: str, config: Config) -> str:
    """Cache key describing the *feature recipe*, not just the backbone.

    Default settings keep short names (``clip_b32``, ``clip_b32+tiles5``); any
    non-default value that changes the numbers adds a suffix, so two different
    recipes can never share a cache file.
    """
    key = backbone_id
    tiles = config.backbone.tiles
    if tiles.enabled:
        key += f"+tiles{tiles.count}"
        if tiles.max_side != TilesConfig().max_side:
            key += f"x{tiles.max_side}"
    if config.backbone.extra_features.deepghs_real:
        key += "+deepghs"
    background = tuple(config.image.alpha_background)
    if background != DEFAULT_BACKGROUND:
        key += "+bg" + "".join(f"{channel:02x}" for channel in background)
    return key


def feature_recipe(backbone_id: str, config: Config) -> dict:
    """Everything that determines the feature numbers; stored in model bundles."""
    return {
        "backbone": backbone_id,
        "feature_key": feature_key(backbone_id, config),
        "tiles": config.backbone.tiles.model_dump(),
        "deepghs_real": bool(config.backbone.extra_features.deepghs_real),
        "alpha_background": list(config.image.alpha_background),
    }


# --------------------------------------------------------------------------- #
# Cache I/O
# --------------------------------------------------------------------------- #
def embedding_cache_path(config: Config, key: str) -> Path:
    """Main cache file for a feature key (checkpoint shards sit next to it)."""
    return config.embeddings_dir / f"{key}.npz"


def _shard_paths(config: Config, key: str) -> list[Path]:
    return sorted(config.embeddings_dir.glob(f"{key}.part-*.npz"))


def _read_npz(path: Path) -> tuple[list[str], np.ndarray, dict]:
    with np.load(path, allow_pickle=False) as data:
        shas = [str(s) for s in data["sha1"].tolist()]
        feats = np.asarray(data["feats"], dtype=np.float32)
        meta = json.loads(str(data["meta"]))
    return shas, feats, meta


def load_embedding_cache(config: Config, key: str) -> tuple[dict[str, np.ndarray], dict]:
    """Return ``(sha1 -> vector, meta)``; empty mapping if the cache is absent.

    Reads the main file plus any checkpoint shards left by an interrupted run.
    Failures recorded in any file are merged into ``meta["failed"]``.
    """
    mapping: dict[str, np.ndarray] = {}
    meta: dict = {}
    failed: dict[str, str] = {}
    main = embedding_cache_path(config, key)
    files = ([main] if main.exists() else []) + _shard_paths(config, key)
    for path in files:
        shas, feats, file_meta = _read_npz(path)
        for i, sha in enumerate(shas):
            mapping[sha] = feats[i]
        failed.update(file_meta.get("failed", {}))
        if file_meta.get("block_sizes"):
            meta = file_meta
    if meta or failed:
        meta = {**meta, "failed": {sha: err for sha, err in failed.items() if sha not in mapping}}
    return mapping, meta


def load_feature_meta(config: Config, backbone_id: str | None = None) -> dict:
    """Read only the meta of the cache for the configured recipe (cheap)."""
    key = feature_key(backbone_id or config.backbone.id, config)
    for path in [embedding_cache_path(config, key), *_shard_paths(config, key)]:
        if path.exists():
            with np.load(path, allow_pickle=False) as data:
                meta = json.loads(str(data["meta"]))
            if meta.get("block_sizes"):
                return meta
    return {}


def _write_npz(path: Path, mapping: dict[str, np.ndarray], meta: dict) -> None:
    shas = list(mapping.keys())
    if shas:
        feats = np.stack([mapping[sha] for sha in shas]).astype(np.float32)
    else:
        feats = np.empty((0, 0), dtype=np.float32)
    # Write to a temp file then rename, so a crash never leaves a torn cache.
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, sha1=np.array(shas, dtype=str), feats=feats, meta=np.array(json.dumps(meta)))
    tmp.replace(path)


def _save_embedding_cache(config: Config, key: str, mapping: dict[str, np.ndarray], meta: dict) -> None:
    """Write the full cache and remove checkpoint shards it now contains."""
    config.embeddings_dir.mkdir(parents=True, exist_ok=True)
    _write_npz(embedding_cache_path(config, key), mapping, meta)
    for shard in _shard_paths(config, key):
        shard.unlink(missing_ok=True)


class _Checkpointer:
    """Appends only the *new* vectors as small shard files.

    Rewriting the whole cache on every checkpoint costs O(N) each time and
    O(N²) over a long run; shards make each checkpoint proportional to the
    work done since the last one.
    """

    def __init__(self, config: Config, key: str, meta: dict) -> None:
        self.config = config
        self.key = key
        self.meta = meta
        self.pending: dict[str, np.ndarray] = {}
        self.failed: dict[str, str] = {}

    def add(self, sha: str, vector: np.ndarray) -> None:
        """Queue a freshly computed vector for the next checkpoint shard."""
        self.pending[sha] = vector

    def fail(self, sha: str, error: str) -> None:
        """Remember an image whose bytes cannot be decoded, so it is not retried."""
        self.failed[sha] = error

    def flush(self) -> None:
        """Write queued vectors/failures as a new shard and clear the queue."""
        if not self.pending and not self.failed:
            return
        self.config.embeddings_dir.mkdir(parents=True, exist_ok=True)
        index = len(_shard_paths(self.config, self.key))
        path = self.config.embeddings_dir / f"{self.key}.part-{index:05d}.npz"
        _write_npz(path, self.pending, {**self.meta, "failed": self.failed})
        self.pending, self.failed = {}, {}


# --------------------------------------------------------------------------- #
# DataLoader plumbing
# --------------------------------------------------------------------------- #
class _ImageDataset(Dataset):
    """Loads + preprocesses one image per item; failures are returned, not raised.

    ``preprocess`` is a picklable object (see ``backbones.base``) and the dataset
    holds no model, so this is safe with ``num_workers > 0`` under spawn (G9).
    """

    def __init__(self, items: list[tuple[int, str]], preprocess, bg, max_pixels: int) -> None:
        self.items = items
        self.preprocess = preprocess
        self.bg = bg
        self.max_pixels = max_pixels

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        item_index, path = self.items[index]
        if not Path(path).is_file():
            # Missing file: transient (moved/unmounted), so not remembered.
            return item_index, None, ("missing file", False)
        try:
            image = load_rgb(Path(path), self.bg, self.max_pixels)
            return item_index, self.preprocess(image), None
        except Exception as exc:  # noqa: BLE001 - report and skip
            # The bytes themselves are undecodable: same sha1, same result, so
            # the failure is remembered and not retried on every call.
            return item_index, None, (f"{type(exc).__name__}: {exc}", True)


def _collate(batch):
    """Stack the images that loaded; return failed indices separately (G3)."""
    indices, tensors, failures = [], [], []
    for index, tensor, error in batch:
        if tensor is None:
            failures.append((index, *error))
        else:
            indices.append(index)
            tensors.append(tensor)
    stacked = torch.stack(tensors) if tensors else torch.empty(0)
    return indices, stacked, failures


# --------------------------------------------------------------------------- #
# Tiles, extra block, device fallback
# --------------------------------------------------------------------------- #
def _tile_crops(image: Image.Image, image_size: int, count: int, max_side: int, seed: int):
    """Center crop plus random native-resolution crops (G24)."""
    if max(image.size) > max_side:
        scale = max_side / max(image.size)
        new_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        image = image.resize(new_size, Image.BILINEAR)
    width, height = image.size
    crop_w, crop_h = min(image_size, width), min(image_size, height)
    rng = np.random.default_rng(seed)
    crops = [image.crop(((width - crop_w) // 2, (height - crop_h) // 2,
                         (width - crop_w) // 2 + crop_w, (height - crop_h) // 2 + crop_h))]
    for _ in range(count - 1):
        x = int(rng.integers(0, width - crop_w + 1)) if width > crop_w else 0
        y = int(rng.integers(0, height - crop_h + 1)) if height > crop_h else 0
        crops.append(image.crop((x, y, x + crop_w, y + crop_h)))
    return crops


def _sha_seed(sha: str) -> int:
    """Deterministic per-image RNG seed, so tile positions never change."""
    return int(sha[:8], 16)


def _extra_block(path: str) -> np.ndarray:
    """deepghs P(real) as a single feature, mapped to ``2p - 1`` in [-1, 1]."""
    from .backbones.deepghs import real_score

    return np.array([2.0 * real_score(path) - 1.0], dtype=np.float32)


class _Encoder:
    """Runs ``backbone.encode`` and falls back to CPU if an MPS op fails.

    Some ops are missing on Apple's MPS backend. Rather than crash a long run,
    the backbone is rebuilt on CPU once and the failure is logged (pitfall 4).
    """

    def __init__(self, backbone) -> None:
        self.backbone = backbone

    def __call__(self, batch: torch.Tensor) -> np.ndarray:
        try:
            return self.backbone.encode(batch).float().cpu().numpy()
        except (RuntimeError, NotImplementedError) as exc:
            if getattr(self.backbone, "_device", "cpu") != "mps":
                raise
            log.warning("MPS failed for %s (%s); falling back to CPU.", self.backbone.id, exc)
            self.backbone = build_backbone(self.backbone.id, "cpu")
            return self.backbone.encode(batch.cpu()).float().cpu().numpy()


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def get_features(
    config: Config,
    shas: list[str],
    paths: list[str],
    backbone_id: str | None = None,
    return_mask: bool = False,
    retry_failed: bool = False,
):
    """Return an ``(N, D)`` float32 matrix of features, computing only what's missing.

    With ``return_mask=True`` the result is ``(matrix, valid)`` where ``valid``
    is a boolean array; rows for images that could not be embedded are zero and
    ``valid`` is False. Callers that train, evaluate or predict must drop them.
    ``retry_failed`` re-attempts images that failed before (``pvd embed`` does).
    """
    if len(shas) != len(paths):
        raise ValueError("shas and paths must have the same length")
    backbone_id = backbone_id or config.backbone.id
    key = feature_key(backbone_id, config)
    mapping, meta = load_embedding_cache(config, key)
    failed: dict[str, str] = {} if retry_failed else dict(meta.get("failed", {}))
    if config.backbone.extra_features.deepghs_real:
        from .backbones.deepghs import require_deepghs

        require_deepghs()  # fail before any work, not once per image

    # One job per distinct sha1 not yet cached or known to fail.
    first_index: dict[str, int] = {}
    for i, sha in enumerate(shas):
        if sha not in mapping and sha not in failed and sha not in first_index:
            first_index[sha] = i
    missing = list(first_index.values())

    if missing:
        device = pick_device(config.device)
        backbone = build_backbone(backbone_id, device)
        meta = _build_meta(backbone, key, config)
        checkpoint = _Checkpointer(config, key, meta)
        if config.backbone.tiles.enabled:
            _embed_with_tiles(config, backbone, checkpoint, mapping, shas, paths, missing)
        else:
            _embed_global(config, backbone, checkpoint, mapping, shas, paths, missing)
        failed.update(checkpoint.failed)
        checkpoint.flush()
        _save_embedding_cache(config, key, mapping, {**meta, "failed": failed})
    elif len(_shard_paths(config, key)) > MAX_SHARDS:
        _save_embedding_cache(config, key, mapping, meta)

    dim = int(sum(meta.get("block_sizes", {}).values())) or _feature_dim_from_mapping(mapping)
    matrix = np.zeros((len(shas), dim), dtype=np.float32)
    valid = np.zeros(len(shas), dtype=bool)
    for i, sha in enumerate(shas):
        vector = mapping.get(sha)
        if vector is not None:
            matrix[i] = vector
            valid[i] = True
    n_invalid = int((~valid).sum())
    if n_invalid:
        log.warning(
            "%d image(s) have no embedding and are excluded (see `pvd embed` output).",
            n_invalid,
        )
    return (matrix, valid) if return_mask else matrix


def global_block(features: np.ndarray, meta: dict) -> np.ndarray:
    """The text-aligned, unit-norm global block of a feature matrix.

    Zero-shot and cosine similarity must use only this block: the tile block
    lives in the same space but is a different view, and concatenating blocks
    makes row norms ≈ √2 so dot products stop being cosines.
    """
    dim = int(meta.get("block_sizes", {}).get("global", features.shape[1]))
    return features[:, :dim]


def _feature_dim_from_mapping(mapping: dict[str, np.ndarray]) -> int:
    if not mapping:
        return 0
    return int(next(iter(mapping.values())).shape[0])


def _versions() -> dict:
    versions = {}
    try:
        import open_clip

        versions["open_clip"] = open_clip.__version__
    except Exception:  # pragma: no cover
        pass
    try:
        import sklearn

        versions["sklearn"] = sklearn.__version__
    except Exception:  # pragma: no cover
        pass
    return {"torch": torch.__version__, **versions}


def _build_meta(backbone, key: str, config: Config) -> dict:
    """Cache meta: backbone identity, recipe and block layout (G8, NF4)."""
    tiles = config.backbone.tiles
    deepghs = config.backbone.extra_features.deepghs_real
    return {
        **backbone.meta(),
        "feature_key": key,
        # Global and tile blocks are each L2-normalized; the extra block is 2p-1.
        "normalization": "l2",
        "extra_transform": "2*p_real - 1",
        "tiles": tiles.model_dump(),
        "extra_features": config.backbone.extra_features.model_dump(),
        "alpha_background": list(config.image.alpha_background),
        "block_sizes": {
            "global": backbone.dim,
            "tiles": backbone.dim if tiles.enabled else 0,
            "extra": 1 if deepghs else 0,
        },
        "versions": _versions(),
    }


def _embed_global(config, backbone, checkpoint, mapping, shas, paths, missing) -> None:
    """Batched path: DataLoader workers decode, the backbone encodes."""
    use_deepghs = config.backbone.extra_features.deepghs_real
    items = [(i, paths[i]) for i in missing]
    dataset = _ImageDataset(
        items,
        backbone.preprocess,
        tuple(config.image.alpha_background),
        config.image.max_pixels,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.backbone.batch_size,
        num_workers=config.num_workers,
        collate_fn=_collate,
        shuffle=False,
    )
    encode = _Encoder(backbone)
    progress = tqdm(loader, total=len(loader), desc=f"embed {checkpoint.key}", unit="batch")
    for batch_number, (indices, batch, failures) in enumerate(progress, start=1):
        for index, error, permanent in failures:
            log.error("Failed to embed %s: %s", paths[index], error)
            if permanent:
                checkpoint.fail(shas[index], error)
        if len(indices):
            feats = encode(batch)
            for row, index in enumerate(indices):
                vector = feats[row]
                if use_deepghs:
                    try:
                        vector = np.concatenate([vector, _extra_block(paths[index])])
                    except Exception as exc:  # noqa: BLE001 - may be transient
                        log.error("deepghs failed on %s: %s", paths[index], exc)
                        continue
                vector = vector.astype(np.float32)
                mapping[shas[index]] = vector
                checkpoint.add(shas[index], vector)
        if batch_number % CHECKPOINT_EVERY_BATCHES == 0:
            checkpoint.flush()


def _embed_with_tiles(config, backbone, checkpoint, mapping, shas, paths, missing) -> None:
    """Per-image path when tiling is enabled; slower but keeps blocks separate."""
    tiles = config.backbone.tiles
    use_deepghs = config.backbone.extra_features.deepghs_real
    bg = tuple(config.image.alpha_background)
    encode = _Encoder(backbone)
    for done, index in enumerate(tqdm(missing, desc=f"embed {checkpoint.key}", unit="img"), start=1):
        sha, path = shas[index], paths[index]
        if not Path(path).is_file():
            log.error("Failed to embed %s: missing file", path)
            continue
        try:
            image = load_rgb(Path(path), bg, config.image.max_pixels)
            global_vec = encode(backbone.preprocess(image).unsqueeze(0))[0]
            crops = _tile_crops(image, backbone.image_size, tiles.count, tiles.max_side, _sha_seed(sha))
            crop_batch = torch.stack([backbone.preprocess(crop) for crop in crops])
            tile_vec = encode(crop_batch).mean(axis=0)
            tile_vec = tile_vec / max(float(np.linalg.norm(tile_vec)), 1e-12)
            blocks = [global_vec, tile_vec]
            if use_deepghs:
                blocks.append(_extra_block(path))
            vector = np.concatenate(blocks).astype(np.float32)
            mapping[sha] = vector
            checkpoint.add(sha, vector)
        except Exception as exc:  # noqa: BLE001
            log.error("Failed to embed %s: %s", path, exc)
            checkpoint.fail(sha, f"{type(exc).__name__}: {exc}")
        if done % CHECKPOINT_EVERY_IMAGES == 0:
            checkpoint.flush()


def run_embed(config: Config, backbone_id: str | None = None) -> Path:
    """Fill the embedding cache for all ``ok`` manifest rows.

    Images that fail to embed are marked ``broken`` in the manifest so every
    later command skips them (§9.5).
    """
    manifest = load_manifest(config)
    usable = manifest[manifest["status"] == "ok"]
    shas = usable["sha1"].tolist()
    paths = [str(config.resolve(p)) for p in usable["path"].tolist()]

    key = feature_key(backbone_id or config.backbone.id, config)
    before, _ = load_embedding_cache(config, key)
    known = sum(1 for sha in set(shas) if sha in before)

    started = time.time()
    get_features(config, shas, paths, backbone_id, retry_failed=True)
    after, meta = load_embedding_cache(config, key)

    failed = meta.get("failed", {})
    newly_broken = manifest["sha1"].isin(failed) & (manifest["status"] == "ok")
    if newly_broken.any():
        manifest.loc[newly_broken, "status"] = "broken"
        if "decode_status" in manifest.columns:
            manifest.loc[newly_broken, "decode_status"] = "broken"
        manifest.loc[newly_broken, "error"] = manifest.loc[newly_broken, "sha1"].map(failed)
        save_manifest(config, manifest)
        log.warning("Marked %d image(s) broken (could not be embedded).", int(newly_broken.sum()))

    log.info(
        "Embeddings: %d cached, %d computed in %.1fs -> %s",
        known,
        len(after) - len(before),
        time.time() - started,
        embedding_cache_path(config, key),
    )
    if meta.get("block_sizes"):
        log.info("Feature blocks: %s", meta.get("block_sizes"))
    return embedding_cache_path(config, key)


def load_feature_matrix(config: Config, key: str | None = None, backbone_id: str | None = None):
    """Load the whole cache as ``(shas, matrix, meta)`` for training/eval."""
    key = key or feature_key(backbone_id or config.backbone.id, config)
    mapping, meta = load_embedding_cache(config, key)
    if not mapping:
        raise FileNotFoundError(
            f"No embeddings for {key}. Run `pvd embed` first (cache: "
            f"{embedding_cache_path(config, key)})."
        )
    shas = list(mapping.keys())
    matrix = np.stack([mapping[sha] for sha in shas]).astype(np.float32)
    return shas, matrix, meta
