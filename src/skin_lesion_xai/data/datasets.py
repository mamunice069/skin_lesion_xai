"""Datasets reading the split CSVs written by scripts/prepare_*.py.

Split CSV columns: image_id, label, label_name, group_id, split.
Image file = <image_root>/<image_id><image_ext>; mask = <mask_root>/<image_id><mask_suffix>.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from skin_lesion_xai.config import DataConfig
from skin_lesion_xai.data.io import read_mask, read_rgb
from skin_lesion_xai.data.splits import load_split_frame


class SkinLesionDataset(Dataset):
    """Returns dicts with keys image (3xHxW float), label (long), image_id, and mask (HxW uint8) if requested."""

    def __init__(
        self,
        frame: pd.DataFrame,
        image_root: Path,
        transform=None,
        image_ext: str = ".jpg",
        mask_root: Path | None = None,
        mask_suffix: str = "_segmentation.png",
        return_mask: bool = False,
    ):
        if return_mask and mask_root is None:
            raise ValueError("return_mask=True needs a mask_root")
        self.frame = frame.reset_index(drop=True)
        self.image_root = Path(image_root)
        self.transform = transform
        self.image_ext = image_ext
        self.mask_root = Path(mask_root) if mask_root else None
        self.mask_suffix = mask_suffix
        self.return_mask = return_mask

    def __len__(self) -> int:
        return len(self.frame)

    @property
    def labels(self) -> np.ndarray:
        return self.frame["label"].to_numpy()

    def __getitem__(self, idx: int) -> dict:
        row = self.frame.iloc[idx]
        image_id = row["image_id"]
        image = read_rgb(self.image_root / f"{image_id}{self.image_ext}")
        mask = read_mask(self.mask_root / f"{image_id}{self.mask_suffix}") if self.return_mask else None
        if mask is not None and mask.shape != image.shape[:2]:
            raise ValueError(f"{image_id}: mask {mask.shape} does not match image {image.shape[:2]}")

        if self.transform is not None:
            out = self.transform(image=image, mask=mask) if mask is not None else self.transform(image=image)
            image, mask = out["image"], out.get("mask")
        else:
            image = torch.from_numpy(image).permute(2, 0, 1)

        item = {"image": image, "label": torch.tensor(int(row["label"]), dtype=torch.long), "image_id": image_id}
        if mask is not None:
            item["mask"] = torch.as_tensor(mask, dtype=torch.uint8)
        return item


def stratified_subsample(frame: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """About n rows with class proportions kept and at least one row per class (for smoke runs)."""
    if n >= len(frame):
        return frame
    share = n / len(frame)
    parts = [g.sample(max(1, round(len(g) * share)), random_state=seed) for _, g in frame.groupby("label")]
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def build_dataset(cfg: DataConfig, split: str, transform=None, return_mask: bool = False,
                  max_samples: int | None = None, seed: int = 42) -> SkinLesionDataset:
    frame = load_split_frame(cfg, split)
    if max_samples is not None:
        frame = stratified_subsample(frame, max_samples, seed)
    return SkinLesionDataset(
        frame,
        image_root=cfg.image_root,
        transform=transform,
        image_ext=cfg.image_ext,
        mask_root=cfg.mask_root,
        mask_suffix=cfg.mask_suffix,
        return_mask=return_mask,
    )
