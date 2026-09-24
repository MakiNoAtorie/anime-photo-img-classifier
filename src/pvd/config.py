"""Validated configuration for the whole project.

Why a pydantic model instead of a plain dict: every module reads the config, and
a typo in ``config.yaml`` should fail loudly and early (at startup) rather than
silently producing a default. The model is the single source of truth for
defaults, so ``config.yaml`` can be minimal.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Default prompt ensembles (G25). Keys must be class names; kept in sync with the
# default ``classes`` below. Several phrasings per class reduce zero-shot noise.
DEFAULT_PROMPTS: dict[str, list[str]] = {
    "photo": [
        "a photograph",
        "a photo taken with a camera",
        "a smartphone photo",
        "a DSLR photograph",
        "a candid photo of real life",
        "a black and white photograph",
    ],
    "drawing": [
        "a drawing",
        "a digital illustration",
        "an anime illustration",
        "a painting",
        "a pencil sketch",
        "a realistic digital painting",
        "concept art",
        "a manga page",
    ],
}

DEFAULT_EXTENSIONS: list[str] = [
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".gif",
    ".tif",
    ".tiff",
    # iPhone photos; decoded only if the optional `heic` extra is installed,
    # otherwise scan reports them as broken instead of silently skipping them.
    ".heic",
    ".heif",
]

# Folder names `pvd predict` writes next to the class folders.
RESERVED_NAMES = {"unsure", "broken"}


class PathsConfig(BaseModel):
    """Project-relative locations of all inputs and outputs (G7).

    Every path is stored relative to ``project_root`` so the project can be moved
    between machines without breaking labels or caches.
    """

    model_config = ConfigDict(extra="forbid")

    pool: str = "data/pool"
    seed: str = "data/seed"
    test: str = "data/test"
    cache: str = "cache"
    labels: str = "labels/events.jsonl"
    models: str = "models"
    reports: str = "reports"


class ImageConfig(BaseModel):
    """How images are decoded before any model sees them (G3-G5)."""

    model_config = ConfigDict(extra="forbid")

    alpha_background: tuple[int, int, int] = (255, 255, 255)
    # Checked from the header before decoding; must be > 0 (0 would disable the
    # guard). Pillow itself refuses anything above ~178 MP, hence the cap.
    max_pixels: int = Field(default=80_000_000, gt=0, le=178_000_000)
    extensions: list[str] = Field(default_factory=lambda: list(DEFAULT_EXTENSIONS))

    @field_validator("extensions")
    @classmethod
    def _normalize_extensions(cls, value: list[str]) -> list[str]:
        return [ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in value]


class TilesConfig(BaseModel):
    """Multi-scale tiling block (G24). Disabled by default for small pools."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    count: int = Field(default=5, ge=1)
    max_side: int = Field(default=1536, ge=64)


class ExtraFeaturesConfig(BaseModel):
    """Non-backbone scalar features appended to the embedding (G26)."""

    model_config = ConfigDict(extra="forbid")

    deepghs_real: bool = False


class BackboneConfig(BaseModel):
    """Which embedding model to use and how to batch it."""

    model_config = ConfigDict(extra="forbid")

    id: str = "clip_b32"
    batch_size: int = Field(default=32, ge=1)
    tiles: TilesConfig = Field(default_factory=TilesConfig)
    extra_features: ExtraFeaturesConfig = Field(default_factory=ExtraFeaturesConfig)


class ZeroshotConfig(BaseModel):
    """Text prompts for the CLIP/SigLIP zero-shot baseline (G25)."""

    model_config = ConfigDict(extra="forbid")

    prompts: dict[str, list[str]] = Field(
        default_factory=lambda: {k: list(v) for k, v in DEFAULT_PROMPTS.items()}
    )


class ProbeConfig(BaseModel):
    """Logistic-regression probe hyperparameters (G13, G27)."""

    model_config = ConfigDict(extra="forbid")

    C_grid: list[float] = Field(default_factory=lambda: [0.01, 0.1, 1.0, 10.0, 100.0])
    cv_folds: int = Field(default=5, ge=2)
    class_weight: str | None = "balanced"
    max_iter: int = Field(default=5000, ge=1)


class ReviewConfig(BaseModel):
    """Active-learning review loop behaviour (G12, G16-G19)."""

    model_config = ConfigDict(extra="forbid")

    batch_size: int = Field(default=24, ge=1)
    random_fraction: float = Field(default=0.25, ge=0.0, le=1.0)
    diversity: bool = True
    undo_depth: int = Field(default=20, ge=0)
    retrain_every_batch: bool = True
    # Full cross-validated retrain every N confirmed batches; in between, the
    # probe is refit with the last chosen C (fast, NF2).
    cv_every: int = Field(default=10, ge=1)


class PredictConfig(BaseModel):
    """Defaults for the ``pvd predict`` command (G22)."""

    model_config = ConfigDict(extra="forbid")

    unsure_threshold: float = Field(default=0.80, gt=0.0, le=1.0)
    mode: Literal["copy", "move", "csv_only"] = "copy"


class FinetuneConfig(BaseModel):
    """Optional timm fine-tuning stage (M9, §9.14)."""

    model_config = ConfigDict(extra="forbid")

    model_name: str = "convnext_tiny.fb_in22k_ft_in1k"
    pretrained: bool = True
    image_size: int = Field(default=384, ge=32)
    epochs: int = Field(default=8, ge=1)
    batch_size: int = Field(default=16, ge=1)
    lr_head: float = 1e-3
    lr_backbone: float = 1e-4
    weight_decay: float = 0.05
    label_smoothing: float = 0.1
    val_fraction: float = Field(default=0.15, gt=0.0, lt=0.5)
    early_stopping_patience: int = Field(default=3, ge=1)


class Config(BaseModel):
    """Top-level validated configuration.

    ``classes`` order defines class indices and therefore the order of columns in
    reports and the ``predictions.csv`` output. ``skip_label`` is a reserved
    pseudo-label and must never collide with a real class (section 4.4).
    """

    model_config = ConfigDict(extra="forbid")

    project_root: Path = Field(default=Path("."), validate_default=True)
    classes: list[str] = Field(default_factory=lambda: ["photo", "drawing"])
    skip_label: str = "skip"

    paths: PathsConfig = Field(default_factory=PathsConfig)
    image: ImageConfig = Field(default_factory=ImageConfig)
    device: Literal["auto", "cuda", "mps", "cpu"] = "auto"
    num_workers: int = Field(default=2, ge=0)

    backbone: BackboneConfig = Field(default_factory=BackboneConfig)
    zeroshot: ZeroshotConfig = Field(default_factory=ZeroshotConfig)
    probe: ProbeConfig = Field(default_factory=ProbeConfig)
    review: ReviewConfig = Field(default_factory=ReviewConfig)
    predict: PredictConfig = Field(default_factory=PredictConfig)
    finetune: FinetuneConfig = Field(default_factory=FinetuneConfig)

    seed: int = 1234

    # -- validation ---------------------------------------------------------
    @field_validator("project_root")
    @classmethod
    def _absolute_root(cls, value: Path) -> Path:
        """Store the root as an absolute path.

        Every relative path in the manifest, label log and reports is computed
        against this root, so it must never stay as a bare ``.``: an absolute
        file path is not "inside" the relative path ``.`` and comparisons fail.
        ``abspath`` (not ``resolve``) is used so symlinked data folders keep
        their in-project names.
        """
        return Path(os.path.abspath(value))

    @field_validator("classes")
    @classmethod
    def _validate_classes(cls, value: list[str]) -> list[str]:
        if len(value) < 2:
            raise ValueError("at least two classes are required")
        if any(not str(name).strip() for name in value):
            raise ValueError("class names must be non-empty")
        if len(set(value)) != len(value):
            raise ValueError(f"duplicate class names: {value}")
        return value

    @model_validator(mode="before")
    @classmethod
    def _default_prompts_follow_classes(cls, data: Any) -> Any:
        """Use built-in prompts only for configured classes that have them.

        Without this, renaming the classes (e.g. ``[real, anime]``) would fail
        validation because the built-in prompts name ``photo``/``drawing``.
        Prompts written explicitly in config.yaml are still validated strictly.
        """
        if not isinstance(data, dict):
            return data
        zeroshot = data.get("zeroshot") or {}
        if isinstance(zeroshot, dict) and "prompts" in zeroshot:
            return data
        classes = data.get("classes") or ["photo", "drawing"]
        prompts = {name: list(DEFAULT_PROMPTS[name]) for name in classes if name in DEFAULT_PROMPTS}
        return {**data, "zeroshot": {**zeroshot, "prompts": prompts}}

    @model_validator(mode="after")
    def _validate_cross_fields(self) -> "Config":
        reserved = RESERVED_NAMES & set(self.classes)
        if reserved:
            raise ValueError(f"class names {sorted(reserved)} are reserved for `pvd predict` folders")
        if self.skip_label in self.classes:
            raise ValueError(
                f"skip_label {self.skip_label!r} must not also be a class; "
                f"classes={self.classes}"
            )
        unknown = set(self.zeroshot.prompts) - set(self.classes)
        if unknown:
            raise ValueError(
                f"zeroshot prompts defined for non-classes: {sorted(unknown)}; "
                f"classes={self.classes}"
            )
        return self

    # -- path helpers -------------------------------------------------------
    def resolve(self, relative: str | Path) -> Path:
        """Return an absolute path for a project-relative path.

        Absolute inputs are returned unchanged so callers can pass CLI paths
        directly without special-casing.
        """
        path = Path(relative)
        if path.is_absolute():
            return path
        return Path(os.path.abspath(self.project_root / path))

    def relative(self, path: str | Path) -> str:
        """Return ``path`` as a project-relative POSIX string (G7).

        Paths outside the project (e.g. a data folder on another drive) cannot
        be made relative; they are kept absolute rather than crashing.
        """
        absolute = Path(os.path.abspath(path))
        try:
            return absolute.relative_to(self.project_root).as_posix()
        except ValueError:
            return absolute.as_posix()

    @property
    def pool_dir(self) -> Path:
        return self.resolve(self.paths.pool)

    @property
    def seed_dir(self) -> Path:
        return self.resolve(self.paths.seed)

    @property
    def test_dir(self) -> Path:
        return self.resolve(self.paths.test)

    @property
    def cache_dir(self) -> Path:
        return self.resolve(self.paths.cache)

    @property
    def labels_file(self) -> Path:
        return self.resolve(self.paths.labels)

    @property
    def models_dir(self) -> Path:
        return self.resolve(self.paths.models)

    @property
    def reports_dir(self) -> Path:
        return self.resolve(self.paths.reports)

    @property
    def manifest_path(self) -> Path:
        return self.cache_dir / "manifest.parquet"

    @property
    def embeddings_dir(self) -> Path:
        return self.cache_dir / "embeddings"

    @property
    def thumbs_dir(self) -> Path:
        return self.cache_dir / "thumbs"

    def to_dict(self) -> dict[str, Any]:
        """JSON/YAML-safe dump used inside model bundles and reports."""
        return self.model_dump(mode="json")


def load_config(path: str | Path = "config.yaml") -> Config:
    """Load and validate ``config.yaml``.

    A missing file is not an error: the built-in defaults are used. This keeps
    ``pvd`` usable in a fresh folder before any config is written.
    """
    config_path = Path(path)
    if not config_path.exists():
        return Config()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{config_path} must contain a YAML mapping at the top level")
    # A relative project_root means "relative to this config file", so
    # `pvd -c some/dir/config.yaml` works from any working directory.
    root = Path(raw.get("project_root", "."))
    if not root.is_absolute():
        raw["project_root"] = config_path.parent.absolute() / root
    return Config.model_validate(raw)


def write_default_config(path: str | Path) -> None:
    """Write the built-in defaults to ``path`` (used by docs/onboarding)."""
    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    defaults = Config().to_dict()
    # Keep the written file portable: "." means "the folder holding this file".
    defaults["project_root"] = "."
    config_path.write_text(
        yaml.safe_dump(defaults, sort_keys=False),
        encoding="utf-8",
    )
