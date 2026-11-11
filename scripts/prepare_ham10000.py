"""Build the HAM10000 split CSV (grouped by lesion_id, stratified by dx).

Checks that every metadata row has an image and a mask before writing anything.
Usage: python scripts/prepare_ham10000.py [--config configs/data/ham10000.yaml] [--force]
"""
import argparse

import pandas as pd

from skin_lesion_xai.config import DataConfig
from skin_lesion_xai.data.splits import assert_no_group_leakage, grouped_split, split_summary
from skin_lesion_xai.utils.logging import get_logger

log = get_logger("prepare_ham10000")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data/ham10000.yaml")
    parser.add_argument("--force", action="store_true", help="overwrite an existing split CSV")
    args = parser.parse_args()

    cfg = DataConfig.from_yaml(args.config)
    out_path = cfg.project_path(cfg.split_csv)
    if out_path.exists() and not args.force:
        raise SystemExit(f"{out_path} exists; every run must use the same split. Pass --force to rebuild it.")

    meta = pd.read_csv(cfg.raw_path(cfg.metadata_csv))
    for col in (cfg.id_col, cfg.label_col, cfg.group_col):
        if col not in meta.columns:
            raise SystemExit(f"column {col!r} missing from metadata; found {list(meta.columns)}")
    if meta[cfg.id_col].duplicated().any():
        raise SystemExit("duplicate image ids in metadata")
    unknown = set(meta[cfg.label_col]) - set(cfg.classes)
    if unknown:
        raise SystemExit(f"labels not in config classes: {unknown}")

    image_root, mask_root = cfg.raw_path(cfg.image_dir), cfg.mask_root
    ids = meta[cfg.id_col]
    missing_img = [i for i in ids if not (image_root / f"{i}{cfg.image_ext}").exists()]
    missing_mask = [i for i in ids if mask_root and not (mask_root / f"{i}{cfg.mask_suffix}").exists()]
    if missing_img or missing_mask:
        raise SystemExit(f"{len(missing_img)} images and {len(missing_mask)} masks missing, "
                         f"e.g. {(missing_img + missing_mask)[:5]}")
    log.info("metadata rows: %d, groups (%s): %d, all images and masks present",
             len(meta), cfg.group_col, meta[cfg.group_col].nunique())

    frame = pd.DataFrame({
        "image_id": ids,
        "label": meta[cfg.label_col].map(cfg.classes.index),
        "label_name": meta[cfg.label_col],
        "group_id": meta[cfg.group_col],
    })
    frame["split"] = grouped_split(frame["label"], frame["group_id"], cfg.split_ratios, cfg.split_folds, cfg.seed)
    assert_no_group_leakage(frame)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_path, index=False)
    summary = split_summary(frame, cfg.classes)
    summary.to_csv(out_path.with_name(f"{cfg.name}_summary.csv"))
    log.info("wrote %s\n%s", out_path, summary.to_string())


if __name__ == "__main__":
    main()
