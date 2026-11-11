"""Score the maps of scripts/explain.py against the ground-truth lesion masks (xai/localization.py).

Usage: python scripts/eval_xai.py --run outputs/runs/<run-name> [--split val] [--random-weights] [--smoke]

Reads <run>/xai_<split>[_random][_smoke]/maps.npz and writes next to it:
  per_image.csv   one row per image x map: iou, dice, iou_auc, bbox_iou, pointing, energy_in_mask,
                  boundary_energy (+ label, pred, correct, mask and border-band area fractions)
  summary.json    means (and n, and the share of all-zero maps, which score 0) per map over all images,
                  correct / incorrect predictions and each class; mean IoU per sweep threshold.
                  The reference maps (ref_centre, ref_uniform) are scored too.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from skin_lesion_xai.config import PROJECT_DIR, DataConfig, XaiConfig
from skin_lesion_xai.data.datasets import build_dataset
from skin_lesion_xai.data.transforms import build_eval_transform
from skin_lesion_xai.utils.logging import get_logger
from skin_lesion_xai.xai import xai_dir
from skin_lesion_xai.xai.localization import METRICS, boundary_band, reference_maps, score_map, upsample

CHUNK = 64  # maps upsampled at a time


def summarise(frame: pd.DataFrame) -> dict:
    out = {"n": int(len(frame)), "empty_share": float(frame["map_empty"].mean()) if len(frame) else None}
    for metric in METRICS:
        values = frame[metric].dropna()
        out[metric] = float(values.mean()) if len(values) else None
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--random-weights", action="store_true")
    p.add_argument("--smoke", action="store_true", help="score the explain.py --max-samples output")
    args = p.parse_args()

    run_dir = Path(args.run)
    if not run_dir.is_absolute():
        run_dir = PROJECT_DIR / run_dir
    xdir = xai_dir(run_dir, args.split, args.random_weights, args.smoke)
    log = get_logger("eval_xai", xdir / "eval_xai.log")
    info = json.loads((xdir / "explain.json").read_text())
    xcfg = XaiConfig(**info["xai_config"])  # the settings the maps were made with
    snap = json.loads((run_dir / "config.json").read_text())
    data_cfg = DataConfig(**snap["data"])
    classes, size, sweep = data_cfg.classes, data_cfg.image_size, np.asarray(xcfg.sweep)

    npz = np.load(xdir / "maps.npz")
    ids, labels, probs = npz["image_ids"].tolist(), npz["labels"], npz["probs"]
    preds = probs.argmax(axis=1)
    keys = [k for k in npz.files if "@" in k]
    maps = {k: npz[k] for k in keys}
    refs = reference_maps(size, xcfg.centre_sigma)

    ds = build_dataset(data_cfg, args.split, build_eval_transform(size), return_mask=True,
                       max_samples=info["max_samples"], seed=snap["train"]["seed"])
    if len(ds) != len(ids):
        raise ValueError(f"{len(ds)} images in the dataset, {len(ids)} in maps.npz")

    rows, sweeps = [], {}
    for start in range(0, len(ids), CHUNK):
        stop = min(start + CHUNK, len(ids))
        up = {k: upsample(maps[k][start:stop], size) for k in keys}
        for i in range(start, stop):
            item = ds[i]
            if item["image_id"] != ids[i]:
                raise ValueError(f"image order differs at {i}: {item['image_id']} vs {ids[i]}")
            mask = item["mask"].numpy().astype(bool)
            band = boundary_band(mask, xcfg.boundary_half_width)
            base = {"image_id": ids[i], "label": classes[labels[i]], "pred": classes[preds[i]],
                    "correct": bool(preds[i] == labels[i]), "mask_frac": float(mask.mean()),
                    "band_frac": float(band.mean())}
            candidates = [(k.split("@")[0], k.split("@")[1], up[k][i - start], True) for k in keys]
            candidates += [(name, "none", m, name != "ref_uniform") for name, m in refs.items()]
            for method, target, heatmap, pointing in candidates:
                s = score_map(heatmap, mask, xcfg.threshold, sweep, band, pointing=pointing)
                sweeps.setdefault(f"{method}@{target}", []).append(s.pop("iou_sweep"))
                rows.append(base | {"method": method, "target": target} | s)
        log.info("scored %d/%d images", stop, len(ids))

    frame = pd.DataFrame(rows)
    frame.to_csv(xdir / "per_image.csv", index=False)
    summary = {"run": info["run"], "split": args.split, "random_weights": info["random_weights"],
               "checkpoint_epoch": info["checkpoint_epoch"], "threshold": xcfg.threshold,
               "sweep_thresholds": xcfg.sweep, "maps": {}}
    for (method, target), group in frame.groupby(["method", "target"], sort=False):
        key = f"{method}@{target}"
        summary["maps"][key] = {
            "all": summarise(group),
            "correct": summarise(group[group["correct"]]),
            "incorrect": summarise(group[~group["correct"]]),
            "per_class": {c: summarise(group[group["label"] == c]) for c in classes},
            "iou_by_threshold": np.nanmean(np.stack(sweeps[key]), axis=0).tolist(),
        }
    (xdir / "summary.json").write_text(json.dumps(summary, indent=2))

    log.info("%-32s %6s %6s %7s %7s %6s %7s %7s %6s", "map (all images)", "IoU", "Dice", "IoU-AUC", "boxIoU",
             "point", "E-mask", "E-band", "empty")
    for key, s in summary["maps"].items():
        a = s["all"]
        fmt = lambda v: "   nan" if v is None else f"{v:6.3f}"
        log.info("%-32s %s %s  %s  %s %s  %s  %s %s", key, fmt(a["iou"]), fmt(a["dice"]), fmt(a["iou_auc"]),
                 fmt(a["bbox_iou"]), fmt(a["pointing"]), fmt(a["energy_in_mask"]), fmt(a["boundary_energy"]),
                 fmt(a["empty_share"]))
    log.info("-> %s", xdir)


if __name__ == "__main__":
    main()
