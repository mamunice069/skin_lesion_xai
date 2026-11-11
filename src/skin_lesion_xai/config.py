"""YAML config loading."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

# skin_lesion_xai/ (the folder holding configs/, scripts/, src/)
PROJECT_DIR = Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_DIR / path
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@dataclass
class DataConfig:
    name: str
    task: str  # "multiclass" or "binary"
    data_root: str
    image_dir: str
    metadata_csv: str
    id_col: str
    label_col: str
    group_col: str
    classes: list[str]
    split_csv: str
    mask_dir: str | None = None
    mask_suffix: str = "_segmentation.png"
    image_ext: str = ".jpg"
    split_ratios: dict[str, float] = field(
        default_factory=lambda: {"train": 0.70, "val": 0.15, "test": 0.15}
    )
    split_folds: int = 20
    seed: int = 42
    hair_removed_dir: str | None = None
    use_hair_removal: bool = False
    dullrazor: dict = field(default_factory=dict)
    image_size: int = 224

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DataConfig":
        # Unknown keys raise TypeError, so typos in the YAML fail loudly.
        return cls(**load_yaml(path))

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def raw_path(self, rel: str) -> Path:
        """Path of raw data (images, metadata, masks), relative to data_root."""
        return (PROJECT_DIR / self.data_root / rel).resolve()

    def project_path(self, rel: str) -> Path:
        """Path of generated files (splits, caches), relative to skin_lesion_xai/."""
        return (PROJECT_DIR / rel).resolve()

    @property
    def image_root(self) -> Path:
        """Folder images are read from: the DullRazor cache if enabled, else the raw folder."""
        if self.use_hair_removal:
            if not self.hair_removed_dir:
                raise ValueError(f"{self.name}: use_hair_removal is set but hair_removed_dir is empty")
            return self.project_path(self.hair_removed_dir)
        return self.raw_path(self.image_dir)

    @property
    def mask_root(self) -> Path | None:
        return self.raw_path(self.mask_dir) if self.mask_dir else None


# fields that must match when a run's saved data config is pointed at another copy of the same data
DATA_IDENTITY_FIELDS = ("name", "task", "classes", "id_col", "label_col", "group_col", "split_csv", "image_size")


def relocate_data_config(saved: dict, path: str | Path) -> DataConfig:
    """The data config of a run (config.json "data") with its file locations taken from another YAML,
    e.g. a run trained on Kaggle evaluated locally. Refuses if the two describe different data."""
    other = load_yaml(path)
    diffs = [f for f in DATA_IDENTITY_FIELDS if saved.get(f) != other.get(f, DataConfig.__dataclass_fields__[f].default)]
    if diffs:
        raise ValueError(f"{path} describes different data than the run ({', '.join(diffs)} differ)")
    locations = ("data_root", "image_dir", "metadata_csv", "mask_dir", "mask_suffix", "image_ext", "hair_removed_dir")
    return DataConfig(**(saved | {k: other[k] for k in locations if k in other}))


@dataclass
class ModelConfig:
    name: str
    # "timm": timm classifier as-is; "single_branch": feature backbone + pooling head;
    # "hybrid": EfficientNetV2 (local) + Swin (global) + fusion (models/hybrid.py)
    arch: str
    timm_name: str = ""  # timm / single_branch
    pretrained: bool = True
    drop_rate: float = 0.2  # dropout before the classifier
    drop_path_rate: float = 0.1  # stochastic depth inside the backbone
    match_pretrained_norm: bool = False  # re-normalise inputs to each checkpoint's own mean/std (backbones.py)
    # hybrid only
    local_timm_name: str = ""
    global_timm_name: str = ""
    fusion: str = "cross_attention"  # or "concat"
    fusion_dim: int = 384
    fusion_layers: int = 2
    fusion_heads: int = 8

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ModelConfig":
        return cls(**load_yaml(path))


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 8
    accum_steps: int = 4  # effective batch = batch_size * accum_steps
    num_workers: int = 2  # train DataLoader workers
    val_num_workers: int = 2  # val/test DataLoader workers (0 = load in the main process)
    backbone_lr: float = 1e-4
    head_lr: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 3
    min_lr_ratio: float = 0.01  # cosine decays to lr * min_lr_ratio
    grad_clip: float = 1.0  # 0 disables
    label_smoothing: float = 0.1
    loss: str = "ce"  # "ce" or "focal"
    focal_gamma: float = 2.0
    class_weight_power: float = 0.5  # loss weights ~ (1/freq)**power; 0 disables
    sampler: str = "none"  # "none" or "balanced"
    ema_decay: float = 0.999  # 0 disables
    early_stop_patience: int = 7
    monitor: str = "auto"  # metric for model selection; auto = macro_f1 (multiclass) / auc (binary)
    amp: bool = True
    grad_checkpointing: bool = False
    seed: int = 42

    @classmethod
    def from_yamls(cls, *paths: str | Path, **overrides) -> "TrainConfig":
        """Later files override earlier ones (e.g. default.yaml, then a hardware profile)."""
        merged: dict = {}
        for path in paths:
            merged.update(load_yaml(path))
        merged.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**merged)

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.accum_steps


@dataclass
class XaiConfig:
    threshold: float = 0.15  # binarise at this fraction of the map's max (Grad-CAM §4.1)
    sweep: list[float] = field(default_factory=lambda: [round(0.05 * i, 2) for i in range(1, 20)])
    boundary_half_width: int = 5  # px at 224 px, each side of the lesion border
    centre_sigma: float = 0.25  # std of the centre-Gaussian reference map, as a fraction of image size
    figures_per_class: int = 2
    figure_seed: int = 0
    batch_size: int = 16

    @classmethod
    def from_yaml(cls, path: str | Path) -> "XaiConfig":
        return cls(**load_yaml(path))


@dataclass
class TrustConfig:
    """Settings of the M4 explanation-trust metrics (scripts/eval_trust.py, scripts/compare_xai_paired.py)."""
    maps: dict[str, list[str]] = field(default_factory=dict)  # model name -> Grad-CAM maps to score
    steps: int = 50  # deletion / insertion steps of 1/steps of the pixels each
    blur_kernel: int = 11  # insertion starts from this Gaussian blur of the image
    blur_sigma: float = 5.0
    random_grid: int = 7  # random reference map: uniform noise on this grid, upsampled like the real maps
    centre_sigma: float = 0.25
    noise_std: float = 0.03  # stability: Gaussian noise, std on the 0-1 pixel scale
    brightness: list[float] = field(default_factory=lambda: [0.9, 1.1])
    stability_unchanged_only: bool = True  # average only perturbations that left the prediction unchanged
    hog_pixels_per_cell: int = 16
    seed: int = 0
    batch_size: int = 16
    forward_batch: int = 64  # perturbed images per forward pass in the deletion / insertion curves
    n_boot: int = 1000
    primary: list[str] = field(default_factory=list)  # endpoints Holm-corrected in compare_xai_paired.py
    # Part B (exploratory) explanation methods, xai/cam_variants.py; the defaults reproduce the M4 Grad-CAM run
    method: str = "gradcam"  # "gradcam", "smoothgrad" or "scorecam"
    smoothgrad_samples: int = 50
    smoothgrad_spread: float = 0.15  # noise std as a fraction of each input's (max - min)
    randomisation: str = "cascade"  # "cascade" (block by block) or "full" (every cascade block at once)
    max_samples: int | None = None  # fixed stratified subset of the split (seeded by the run's seed); None = all
    localisation: bool = False  # score IoU / energy in mask / pointing here (configs/xai/default.yaml protocol)
    tag: str | None = None  # output folder suffix instead of the method name (trust_<split>_<tag>)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrustConfig":
        return cls(**load_yaml(path))
