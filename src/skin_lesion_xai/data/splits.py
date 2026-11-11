"""Leakage-free train/val/test splits.

Images are grouped (HAM10000: lesion_id, ISIC 2020: patient_id) so the same lesion or
patient never appears in two splits. Both GC-ViT-E (§3.1) and Mahesh et al. (§4.1) split
by image, which lets near-duplicate images of one lesion leak into the test set.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SPLITS = ("train", "val", "test")


def grouped_split(
    labels: pd.Series,
    groups: pd.Series,
    ratios: dict[str, float],
    n_folds: int = 20,
    seed: int = 42,
) -> pd.Series:
    """Assign each row to train/val/test, stratified by label and grouped by group.

    The data is cut into `n_folds` stratified group folds; test and val take
    ratio * n_folds folds each, train takes the rest.
    """
    # imported here: sklearn pulls in scipy, too heavy for the hair-removal worker processes
    from sklearn.model_selection import StratifiedGroupKFold

    if abs(sum(ratios[s] for s in SPLITS) - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to 1, got {ratios}")
    n_fold = {s: round(ratios[s] * n_folds) for s in ("val", "test")}
    for s, k in n_fold.items():
        if abs(k - ratios[s] * n_folds) > 1e-6:
            raise ValueError(f"ratio {s}={ratios[s]} is not a multiple of 1/{n_folds}")

    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_of = np.empty(len(labels), dtype=int)
    for fold, (_, idx) in enumerate(sgkf.split(np.zeros(len(labels)), labels.values, groups.values)):
        fold_of[idx] = fold

    split = np.full(len(labels), "train", dtype=object)
    split[fold_of < n_fold["test"]] = "test"
    split[(fold_of >= n_fold["test"]) & (fold_of < n_fold["test"] + n_fold["val"])] = "val"
    return pd.Series(split, index=labels.index, name="split")


def assert_no_group_leakage(frame: pd.DataFrame, group_col: str = "group_id") -> None:
    """Raise if any group id appears in more than one split."""
    n_splits = frame.groupby(group_col)["split"].nunique()
    leaked = n_splits[n_splits > 1]
    if len(leaked):
        raise AssertionError(f"{len(leaked)} groups appear in more than one split, e.g. {list(leaked.index[:5])}")


def load_split_frame(cfg, split: str | None = None) -> pd.DataFrame:
    """Read the split CSV of a DataConfig, optionally only one split."""
    path = cfg.project_path(cfg.split_csv)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run scripts/prepare_{cfg.name}.py first")
    frame = pd.read_csv(path)
    return frame if split is None else frame[frame["split"] == split].reset_index(drop=True)


def split_summary(frame: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    """Image counts per class and split (data description, not a result)."""
    table = pd.crosstab(frame["label_name"], frame["split"]).reindex(index=classes, columns=list(SPLITS), fill_value=0)
    table.loc["total images"] = table.sum()
    table.loc["groups (lesions/patients)"] = frame.groupby("split")["group_id"].nunique().reindex(list(SPLITS)).values
    return table
