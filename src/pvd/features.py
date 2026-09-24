"""Embedding computation and content-addressed cache (G2, G24, G26).

Design: embeddings are computed once per (image bytes, backbone/feature config)
pair and stored in ``cache/embeddings/<feature_key>.npz``. Training, sampling and
evaluation then operate on the cached matrix and are fast (NF2).

The cache key is the **SHA-1 of the file bytes**, so renames and moves never
trigger recomputation. Changing the backbone or the tile settings changes the
``feature_key``, so caches for different feature recipes can never be mixed (G8).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .backbones import build_backbone
from .config import Config
from .device import pick_device
from .imageio import load_rgb
from .scan import load_manifest

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Feature key and cache I/O
# --------------------------------------------------------------------------- #
def feature_key(backbone_id: str, config: Config) -> str:
    """Cache key describing the *feature recipe*, not just the backbone.

    Tile and extra-feature settings are part of the identity so different recipes
    never share a cache file.
    """
    key = backbone_id
    if config.backbone.tiles.enabled:
        key += f"+tiles{config.backbone.tiles.count}"
    if config.backbone.extra_features.deepghs_real:
        key += "+deepghs"
    return key


def embedding_cache_path(config: Config, key: str) -> Path:
    return config.embeddings_dir / f"{key}.npz"


def load_embedding_cache(config: Config, key: str) -> tuple[dict[str, np.ndarray], dict]:
    """Return ``(sha1 -> vector, meta)``; empty mapping if the cache is absent."""
    path = embedding_cache_path(config, key)
    if not path.exists():
        return {}, {}
    with np.load(path, allow_pickle=False) as data:
        shas = [str(s) for s in data["sha1"].tolist()]
        feats = data["feats"]
        meta = json.loads(str(data["meta"]))
    return {sha: feats[i] for i, sha in enumerate(shas)}, meta


def _save_embedding_cache(
    config: Config, key: str, mapping: dict[str, np.ndarray], meta: dict
) -> None:
    config.embeddings_dir.mkdir(parents=True, exist_ok=True)
    path = embedding_cache_path(config, key)
    if mapping:
        shas = list(mapping.keys())
        feats = np.stack([mapping[sha] for sha in shas]).astype(np.float32)
    else:
        shas, feats = [], np.empty((0, 0), dtype=np.float32)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(
        tmp,
        sha1=np.array(shas, dtype=str),
        feats=feats,
        meta=np.array(json.dumps(meta)),
    )
    tmp.replace(path)


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
        try:
            image = load_rgb(Path(path), self.bg, self.max_pixels)
            return item_index, self.preprocess(image), None
        except Exception as exc:  # noqa: BLE001 - report and skip
            return item_index, None, f"{type(exc).__name__}: {exc}"


def _collate(batch):
    indices, tensors, failures = [], [], []
    for index, tensor, error in batch:
        if tensor is None:
            failures.append((index, error))
        else:
            indices.append(index)
            tensors.append(tensor)
    stacked = torch.stack(tensors) if tensors else torch.empty(0)
    return indices, stacked, failures


# --------------------------------------------------------------------------- #
# Tiles and extra blocks
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
    return int(sha[:8], 16)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def get_features(
    config: Config,
    shas: list[str],
    paths: list[str],
    backbone_id: str | None = None,
) -> np.ndarray:
    """Return an ``(N, D)`` float32 matrix of normalized features.

    Only SHA-1s missing from the cache are computed. Failed images are logged and
    filled with zero vectors (scan should already have excluded broken files).
    """
    if len(shas) != len(paths):
        raise ValueError("shas and paths must have the same length")
    backbone_id = backbone_id or config.backbone.id
    key = feature_key(backbone_id, config)
    mapping, meta = load_embedding_cache(config, key)

    missing = [i for i, sha in enumerate(shas) if sha not in mapping]
    if missing:
        device = pick_device(config.device)
        backbone = build_backbone(backbone_id, device)
        tiles = config.backbone.tiles
        if tiles.enabled:
            _embed_with_tiles(config, backbone, key, mapping, shas, paths, missing)
        else:
            _embed_global(config, backbone, key, mapping, shas, paths, missing)
        # (Re)write meta so downstream can validate the recipe and dim (G8).
        meta = {
            **backbone.meta(),
            "feature_key": key,
            "normalization": "l2",
            "tiles": tiles.model_dump(),
            "extra_features": config.backbone.extra_features.model_dump(),
            "block_sizes": {
                "global": backbone.dim,
                "tiles": backbone.dim if tiles.enabled else 0,
                "extra": 1 if config.backbone.extra_features.deepghs_real else 0,
            },
            "versions": _versions(),
        }
        _save_embedding_cache(config, key, mapping, meta)

    dim = _feature_dim_from_mapping(mapping)
    matrix = np.zeros((len(shas), dim), dtype=np.float32)
    failures = 0
    for i, sha in enumerate(shas):
        vector = mapping.get(sha)
        if vector is None:
            failures += 1
        else:
            matrix[i] = vector
    if failures:
        log.warning("%d images had no embedding (see earlier errors)", failures)
    return matrix


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


def _decode_failures(failures, shas: list[str]) -> None:
    for index, error in failures:
        log.error("Failed to embed %s: %s", shas[index], error)


def _embed_global(config, backbone, key, mapping, shas, paths, missing) -> None:
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
    save_every = 10
    for batch_number, (indices, batch, failures) in enumerate(loader, start=1):
        _decode_failures(failures, shas)
        if len(indices):
            feats = backbone.encode(batch).cpu().numpy().astype(np.float32)
            for row, index in enumerate(indices):
                mapping[shas[index]] = feats[row]
        if batch_number % save_every == 0:
            _save_embedding_cache(config, key, mapping, {"feature_key": key, "partial": True})
    _save_embedding_cache(config, key, mapping, {"feature_key": key, "partial": True})


def _embed_with_tiles(config, backbone, key, mapping, shas, paths, missing) -> None:
    """Per-image path when tiling is enabled; slower but keeps blocks separate."""
    tiles = config.backbone.tiles
    use_deepghs = config.backbone.extra_features.deepghs_real
    bg = tuple(config.image.alpha_background)
    for index in missing:
        sha, path = shas[index], paths[index]
        try:
            image = load_rgb(Path(path), bg, config.image.max_pixels)
            global_vec = backbone.encode(
                backbone.preprocess(image).unsqueeze(0)
            )[0].cpu().numpy()
            crops = _tile_crops(
                image, backbone.image_size, tiles.count, tiles.max_side, _sha_seed(sha)
            )
            crop_batch = torch.stack([backbone.preprocess(crop) for crop in crops])
            tile_vec = backbone.encode(crop_batch).mean(dim=0).cpu().numpy()
            tile_vec = tile_vec / max(np.linalg.norm(tile_vec), 1e-12)
            blocks = [global_vec, tile_vec]
            if use_deepghs:
                from .backbones.deepghs import real_score

                blocks.append(np.array([real_score(path)], dtype=np.float32))
            mapping[sha] = np.concatenate(blocks).astype(np.float32)
        except Exception as exc:  # noqa: BLE001
            log.error("Failed to embed %s: %s", path, exc)
    _save_embedding_cache(config, key, mapping, {"feature_key": key, "partial": True})


def run_embed(config: Config, backbone_id: str | None = None) -> Path:
    """Fill the embedding cache for all ``ok`` manifest rows."""
    manifest = load_manifest(config)
    usable = manifest[manifest["status"] == "ok"]
    shas = usable["sha1"].tolist()
    paths = [str(config.resolve(p)) for p in usable["path"].tolist()]

    key = feature_key(backbone_id or config.backbone.id, config)
    before, _ = load_embedding_cache(config, key)
    known = sum(1 for sha in shas if sha in before)

    started = time.time()
    get_features(config, shas, paths, backbone_id)
    after, meta = load_embedding_cache(config, key)

    log.info(
        "Embeddings: %d cached, %d computed in %.1fs -> %s",
        known,
        len(after) - len(before),
        time.time() - started,
        embedding_cache_path(config, key),
    )
    if meta.get("feature_key"):
        log.info("Feature dim: %s", meta.get("block_sizes"))
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
