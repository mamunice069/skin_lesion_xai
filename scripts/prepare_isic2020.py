"""Build the ISIC 2020 split CSV (grouped by patient_id, stratified by target).

Usage: python scripts/prepare_isic2020.py [--config configs/data/isic2020.yaml] [--force]
"""
import argparse

import pandas as pd

from skin_lesion_xai.config import DataConfig
from skin_lesion_xai.data.splits import assert_no_group_leakage, grouped_split, split_summary
from skin_lesion_xai.utils.logging import get_logger

log = get_logger("prepare_isic2020")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data/isic2020.yaml")
    parser.add_argument("--force", action="store_true", help="overwrite an existing split CSV")
    args = parser.parse_args()

    cfg = DataConfig.from_yaml(args.config)
    out_path = cfg.project_path(cfg.split_csv)
    if out_path.exists() and not args.force:
        raise SystemExit(f"{out_path} exists; every run must use the same split. Pass --force to rebuild it.")

    meta = pd.read_csv(cfg.raw_path(cfg.metadata_csv))
    meta = meta.loc[:, ~meta.columns.str.startswith("Unnamed")]  # drop the saved pandas index column
    for col in (cfg.id_col, cfg.label_col, cfg.group_col):
        if col not in meta.columns:
            raise SystemExit(f"column {col!r} missing from metadata; found {list(meta.columns)}")
    if meta[cfg.id_col].duplicated().any():
        raise SystemExit("duplicate image ids in metadata")
    if meta[cfg.group_col].isna().any():
        raise SystemExit(f"{meta[cfg.group_col].isna().sum()} rows have no {cfg.group_col}")
    if not set(meta[cfg.label_col]) <= set(range(cfg.num_classes)):
        raise SystemExit(f"unexpected target values: {sorted(set(meta[cfg.label_col]))}")

    image_root = cfg.raw_path(cfg.image_dir)
    missing = [i for i in meta[cfg.id_col] if not (image_root / f"{i}{cfg.image_ext}").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} images missing, e.g. {missing[:5]}")
    log.info("metadata rows: %d, patients: %d, all images present", len(meta), meta[cfg.group_col].nunique())

    frame = pd.DataFrame({
        "image_id": meta[cfg.id_col],
        "label": meta[cfg.label_col].astype(int),
        "label_name": meta[cfg.label_col].map(dict(enumerate(cfg.classes))),
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
