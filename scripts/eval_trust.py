"""Explanation-trust metrics for one trained run (Milestone 4): faithfulness, stability, parameter randomisation.

Usage: python scripts/eval_trust.py --run outputs/runs/<run-name> [--split val] [--max-samples N]

Loads best.pt strictly and scores the maps listed for the model in the trust config (predicted class, 7x7
grids upsampled to the image), plus the reference maps ref_random and ref_centre. The config's `method`
picks Grad-CAM (configs/xai/trust.yaml, M4) or a Part B variant (trust_smoothgrad.yaml, trust_scorecam.yaml;
xai/cam_variants.py); `max_samples` there fixes a stratified subset, `randomisation: full` randomises every
cascade block in one step, and `localisation: true` adds IoU / energy in mask / pointing (eval_xai.py protocol).
Image seeds (random map, stability noise, SmoothGrad noise) come from the image's position in the full split,
so a subset run uses the same seeds per image as a full run.
  deletion / insertion AUC (xai/faithfulness.py)              every map, reference maps included
  stability: Spearman / SSIM to the map of a flipped, noisy    Grad-CAM maps; target = original prediction
    or brighter / darker image (xai/stability.py)
  cascading randomisation: Spearman / SSIM / HOG Spearman to   Grad-CAM maps; the model is randomised in place
    the trained map after each step (xai/randomization.py)     after the other metrics, top block first
Writes <run>/trust_<split>[_<method>][_smoke]/: per_image.csv (one row per image x map), summary.json (means,
mean deletion / insertion curves, seconds per phase, settings), eval_trust.log. Without `localisation`, the
localisation scores come from scripts/eval_xai.py (xai_<split>/per_image.csv); compare_xai_paired.py joins them.
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from skin_lesion_xai.config import PROJECT_DIR, DataConfig, ModelConfig, TrustConfig, XaiConfig
from skin_lesion_xai.data.datasets import build_dataset, load_split_frame
from skin_lesion_xai.data.transforms import build_eval_transform
from skin_lesion_xai.models.factory import build_model
from skin_lesion_xai.utils.checkpoint import load_checkpoint
from skin_lesion_xai.utils.device import get_device, gpu_memory_summary
from skin_lesion_xai.utils.logging import get_logger
from skin_lesion_xai.utils.seed import seed_everything
from skin_lesion_xai.xai import trust_dir
from skin_lesion_xai.xai.cam_variants import build_explainer
from skin_lesion_xai.xai.faithfulness import curve_auc, perturbation_curves
from skin_lesion_xai.xai.gradcam import freeze
from skin_lesion_xai.xai.localization import boundary_band, centre_gaussian, score_map, upsample

LOCALISATION = ("iou", "energy_in_mask", "pointing")
from skin_lesion_xai.xai.randomization import cascade, randomize_step
from skin_lesion_xai.xai.similarity import similarities
from skin_lesion_xai.xai.stability import perturb, perturbation_names, restore_map


def random_maps(indices: list[int], grid: int, seed: int) -> np.ndarray:
    """One fixed uniform-noise grid per image (seeded by its position in the split)."""
    return np.stack([np.random.default_rng([seed, i]).random((grid, grid)).astype(np.float32) for i in indices])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--trust-config", default="configs/xai/trust.yaml")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-samples", type=int, default=None,
                   help="stratified subset for smoke checks; writes to trust_<split>_smoke/")
    args = p.parse_args()

    run_dir = Path(args.run)
    if not run_dir.is_absolute():
        run_dir = PROJECT_DIR / run_dir
    snap = json.loads((run_dir / "config.json").read_text())
    data_cfg, model_cfg = DataConfig(**snap["data"]), ModelConfig(**snap["model"])
    tcfg = TrustConfig.from_yaml(args.trust_config)
    if model_cfg.name not in tcfg.maps:
        raise SystemExit(f"{args.trust_config} lists no maps for model {model_cfg.name}")
    map_names = tcfg.maps[model_cfg.name]
    out_dir = trust_dir(run_dir, args.split, smoke=args.max_samples is not None, method=tcfg.tag or tcfg.method)
    log = get_logger("eval_trust", out_dir / "eval_trust.log")
    device = get_device()
    seed_everything(snap["train"]["seed"])
    amp = snap["train"]["amp"]

    model = build_model(model_cfg, data_cfg.num_classes, pretrained=False)
    epoch = load_checkpoint(run_dir / "best.pt", model)["epoch"]
    model = freeze(model).to(device).eval()
    explain = build_explainer(tcfg.method, model, map_names, tcfg.smoothgrad_samples, tcfg.smoothgrad_spread,
                              tcfg.forward_batch, amp)
    log.info("loaded %s (epoch %s); method %s, maps %s", run_dir / "best.pt", epoch, tcfg.method, map_names)

    max_samples = args.max_samples if args.max_samples is not None else tcfg.max_samples
    ds = build_dataset(data_cfg, args.split, build_eval_transform(data_cfg.image_size),
                       return_mask=tcfg.localisation, max_samples=max_samples, seed=snap["train"]["seed"])
    split_index = {i: k for k, i in enumerate(load_split_frame(data_cfg, args.split)["image_id"])}
    loader = DataLoader(ds, batch_size=tcfg.batch_size, num_workers=args.num_workers, shuffle=False)
    size = data_cfg.image_size
    centre = centre_gaussian(size, tcfg.centre_sigma)
    perturbations = perturbation_names(tcfg.noise_std, tcfg.brightness)
    xcfg = XaiConfig.from_yaml("configs/xai/default.yaml")
    seconds = defaultdict(float)

    rows: dict[tuple[int, str], dict] = {}  # (image index, map) -> metrics
    trained_raw: dict[str, list[np.ndarray]] = defaultdict(list)  # native grids, for the randomisation step
    curves_sum: dict[str, np.ndarray] = {}
    preds_all = []
    seen = 0
    for bi, batch in enumerate(loader):
        x, y = batch["image"].to(device), batch["label"].numpy()
        seen += len(y)
        idx = [split_index[i] for i in batch["image_id"]]  # position in the full split: seeds and row keys
        seeds = [tcfg.seed * 1_000_003 + i for i in idx]

        t0 = time.perf_counter()
        logits, cams = explain(x, {"pred": None}, seeds)
        pred = logits.argmax(dim=1)
        pred_np = pred.cpu().numpy()
        preds_all.append(pred_np)
        raw = {m: cams["pred"][m] for m in map_names}
        for m in map_names:
            trained_raw[m].append(raw[m])
        up = {m: upsample(raw[m], size) for m in map_names}
        up["ref_random"] = upsample(random_maps(idx, tcfg.random_grid, tcfg.seed), size)
        up["ref_centre"] = np.broadcast_to(centre, (len(y), size, size))
        for m in up:
            for j, i in enumerate(idx):
                rows[(i, m)] = {"image_id": batch["image_id"][j], "label": int(y[j]), "pred": int(pred_np[j]),
                                "correct": bool(pred_np[j] == y[j]), "map": m}
        if tcfg.localisation:
            for j, i in enumerate(idx):
                mask = batch["mask"][j].numpy().astype(bool)
                band = boundary_band(mask, xcfg.boundary_half_width)
                for m in map_names:
                    s = score_map(up[m][j], mask, xcfg.threshold, np.asarray(xcfg.sweep), band)
                    rows[(i, m)] |= {k: s[k] for k in LOCALISATION}
        seconds["maps"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        for name in perturbations:
            xp = perturb(x, name, seeds=seeds)
            logits_p, cams_p = explain(xp, {"pred": pred}, seeds)
            changed = (logits_p.argmax(dim=1) != pred).cpu().numpy()
            for m in map_names:
                restored = upsample(restore_map(cams_p["pred"][m], name), size)
                for j, i in enumerate(idx):
                    s = similarities(up[m][j], restored[j], with_hog=False)
                    rows[(i, m)] |= {f"stab_{name}_spearman": s["spearman"], f"stab_{name}_ssim": s["ssim"],
                                     f"stab_{name}_pred_changed": bool(changed[j])}
        seconds["stability"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        for m, maps in up.items():
            for mode in ("deletion", "insertion"):
                curves = perturbation_curves(model, x, maps, pred, tcfg.steps, mode, tcfg.blur_kernel,
                                             tcfg.blur_sigma, tcfg.forward_batch, amp)
                for j, i in enumerate(idx):
                    rows[(i, m)][f"{mode}_auc"] = float(curve_auc(curves[j:j + 1])[0])
                curves_sum[f"{m}@{mode}"] = curves_sum.get(f"{m}@{mode}", 0) + curves.sum(axis=0)
        seconds["faithfulness"] += time.perf_counter() - t0
        if bi % 10 == 0:
            log.info("batch %d/%d  %s  GPU peak %s", bi + 1, len(loader),
                     "  ".join(f"{k} {v:.0f}s" for k, v in seconds.items()), gpu_memory_summary(device))

    trained = {m: np.concatenate(v) for m, v in trained_raw.items()}
    preds = torch.as_tensor(np.concatenate(preds_all))
    steps = cascade(model)
    if tcfg.randomisation == "full":  # the cascade's end state (every block re-initialised) in one step
        steps = [("full", [p for _, paths in steps for p in paths])]
    elif tcfg.randomisation != "cascade":
        raise SystemExit(f"randomisation must be 'cascade' or 'full', not {tcfg.randomisation}")
    for step, paths in steps:  # cumulative: each step randomises one more block, top down
        t0 = time.perf_counter()
        randomize_step(model, paths)
        pos = 0
        for batch in loader:
            x = batch["image"].to(device)
            order = list(range(pos, pos + len(x)))  # position in the loader: index of trained maps / preds
            pos += len(x)
            idx = [split_index[i] for i in batch["image_id"]]
            seeds = [tcfg.seed * 1_000_003 + i for i in idx]
            _, cams = explain(x, {"pred": preds[order].to(device)}, seeds)
            for m in map_names:
                a, b = upsample(trained[m][order], size), upsample(cams["pred"][m], size)
                for j, i in enumerate(idx):
                    s = similarities(a[j], b[j], tcfg.hog_pixels_per_cell)
                    rows[(i, m)] |= {f"rand_{step}_{k}": v for k, v in s.items()}
        seconds[f"randomisation_{step}"] = time.perf_counter() - t0
        log.info("randomised %s (%s): %.0fs", step, ", ".join(paths), seconds[f"randomisation_{step}"])

    last = steps[-1][0]
    frame = pd.DataFrame([rows[k] for k in sorted(rows)])
    for k in ("spearman", "ssim", "hog"):
        frame[f"rand_full_{k}"] = frame[f"rand_{last}_{k}"]
    for k in ("spearman", "ssim"):  # per-perturbation columns keep every image; the mean may drop changed ones
        values = frame[[f"stab_{n}_{k}" for n in perturbations]].astype(float)
        if tcfg.stability_unchanged_only:
            changed = frame[[f"stab_{n}_pred_changed" for n in perturbations]].fillna(False).astype(bool)
            values = values.where(~changed.to_numpy())
        frame[f"stab_{k}_mean"] = values.mean(axis=1)  # NaN if every perturbation changed the prediction
    frame["stab_n_unchanged"] = (~frame[[f"stab_{n}_pred_changed" for n in perturbations]]
                                 .fillna(True).astype(bool)).sum(axis=1)
    frame.insert(1, "label_name", [data_cfg.classes[c] for c in frame["label"]])
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / "per_image.csv", index=False)

    metric_cols = [c for c in frame.columns if c.endswith(("_auc", "_spearman", "_ssim", "_hog", "_mean", "_changed"))]
    metric_cols += [c for c in LOCALISATION if c in frame.columns]
    summary = {"run": snap["run_name"], "split": args.split, "n": int(seen), "checkpoint_epoch": epoch,
               "method": tcfg.method, "max_samples": max_samples, "maps": map_names, "perturbations": perturbations,
               "randomisation_steps": steps, "trust_config": vars(tcfg),
               "seconds": {k: round(v, 1) for k, v in seconds.items()},
               "means": {m: {c: (None if g[c].isna().all() else float(g[c].astype(float).mean())) for c in metric_cols}
                         for m, g in frame.groupby("map", sort=False)},
               "mean_curves": {k: (v / seen).tolist() for k, v in curves_sum.items()}}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    log.info("%-14s %8s %9s %8s %8s %9s %9s", "map", "del AUC", "ins AUC", "stab rho", "stab SSIM",
             "rand rho", "rand SSIM")
    fmt = lambda v: "      -" if v is None else f"{v:8.3f}"
    for m, s in summary["means"].items():
        log.info("%-14s %s  %s %s  %s  %s  %s", m, fmt(s["deletion_auc"]), fmt(s["insertion_auc"]),
                 fmt(s["stab_spearman_mean"]), fmt(s["stab_ssim_mean"]), fmt(s["rand_full_spearman"]),
                 fmt(s["rand_full_ssim"]))
    log.info("seconds: %s", summary["seconds"])
    log.info("-> %s", out_dir)


if __name__ == "__main__":
    main()
