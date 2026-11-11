"""Explanation maps for every image of a split (Grad-CAM, plus attention maps where the model has them).

Usage: python scripts/explain.py --run outputs/runs/<run-name> [--split val] [--random-weights]

Loads best.pt strictly (EMA weights), runs in float32 without gradient checkpointing, and writes to
<run>/xai_<split>/ (xai_<split>_random/ with --random-weights; a _smoke suffix with --max-samples):
  maps.npz      raw maps at their native grid size, one array per "<method>@<target>" (N, h, w),
                plus image_ids, labels, probs; target = pred (predicted class), true (true class) or
                none (class-agnostic attention maps). scripts/eval_xai.py scores these against the masks.
  explain.json  run, split, checkpoint epoch, methods, settings
  figures/      strips for the same images in every run (figures_per_class per class, fixed seed)
--random-weights explains the same architecture with random weights (no checkpoint; zero-initialised
heads re-drawn), the sanity check that localisation scores come from training, not from the method.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from skin_lesion_xai.config import PROJECT_DIR, DataConfig, ModelConfig, XaiConfig, relocate_data_config
from skin_lesion_xai.data.datasets import build_dataset
from skin_lesion_xai.data.transforms import build_eval_transform
from skin_lesion_xai.models.factory import build_model
from skin_lesion_xai.utils.checkpoint import load_checkpoint
from skin_lesion_xai.utils.device import get_device, gpu_memory_summary
from skin_lesion_xai.utils.logging import get_logger
from skin_lesion_xai.utils.seed import seed_everything
from skin_lesion_xai.xai import xai_dir
from skin_lesion_xai.xai.attention_maps import attention_maps, attention_methods
from skin_lesion_xai.xai.gradcam import GradCAM, explain_layers, freeze
from skin_lesion_xai.xai.localization import boundary_band, score_map, upsample
from skin_lesion_xai.xai.visualize import denormalize, figure_indices, save_strip


def redraw_zero_heads(model: nn.Module) -> None:
    """Zero-initialised linear layers (heads, ViT App. B.1.1) give all-zero gradients; draw them randomly."""
    for module in model.modules():
        if isinstance(module, nn.Linear) and not module.weight.any():
            nn.init.trunc_normal_(module.weight, std=0.02)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--xai-config", default="configs/xai/default.yaml")
    p.add_argument("--batch-size", type=int, default=None, help="overrides the XAI config")
    p.add_argument("--num-workers", type=int, default=0, help="0 = load in the main process (low-RAM machines)")
    p.add_argument("--random-weights", action="store_true")
    p.add_argument("--max-samples", type=int, default=None,
                   help="stratified subset for smoke checks; writes to xai_<split>[_random]_smoke/")
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--data", default=None,
                   help="data YAML giving this machine's file locations (e.g. for a run trained on Kaggle)")
    args = p.parse_args()

    run_dir = Path(args.run)
    if not run_dir.is_absolute():
        run_dir = PROJECT_DIR / run_dir
    snap = json.loads((run_dir / "config.json").read_text())
    data_cfg = relocate_data_config(snap["data"], args.data) if args.data else DataConfig(**snap["data"])
    model_cfg = ModelConfig(**snap["model"])
    xcfg = XaiConfig.from_yaml(args.xai_config)
    batch_size = args.batch_size or xcfg.batch_size
    out_dir = xai_dir(run_dir, args.split, args.random_weights, smoke=args.max_samples is not None)
    log = get_logger("explain", out_dir / "explain.log")
    device = get_device()
    seed_everything(snap["train"]["seed"])

    model = build_model(model_cfg, data_cfg.num_classes, pretrained=False)
    if args.random_weights:
        redraw_zero_heads(model)
        epoch = None
        log.info("random weights (no checkpoint loaded)")
    else:
        epoch = load_checkpoint(run_dir / "best.pt", model)["epoch"]
        log.info("loaded %s (epoch %s)", run_dir / "best.pt", epoch)
    model = freeze(model).to(device).eval()
    layers, combine = explain_layers(model)
    cam = GradCAM(model, layers, combine)

    ds = build_dataset(data_cfg, args.split, build_eval_transform(data_cfg.image_size),
                       max_samples=args.max_samples, seed=snap["train"]["seed"])
    loader = DataLoader(ds, batch_size=batch_size, num_workers=args.num_workers, shuffle=False)
    store: dict[str, list[np.ndarray]] = defaultdict(list)
    ids, labels, probs = [], [], []
    for i, batch in enumerate(loader):
        x, y = batch["image"].to(device), batch["label"]
        logits, cams = cam(x, {"pred": None, "true": y})
        for target, maps in cams.items():
            for name, m in maps.items():
                store[f"{name}@{target}"].append(m)
        for name, m in attention_maps(model, x).items():
            store[f"{name}@none"].append(m)
        ids += list(batch["image_id"])
        labels.append(y.numpy())
        probs.append(logits.float().softmax(dim=1).cpu().numpy())
        if i % 20 == 0:
            log.info("batch %d/%d  GPU peak %s", i + 1, len(loader), gpu_memory_summary(device))

    arrays = {key: np.concatenate(parts).astype(np.float32) for key, parts in store.items()}
    labels_arr, probs_arr = np.concatenate(labels), np.concatenate(probs)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "maps.npz", image_ids=np.array(ids), labels=labels_arr, probs=probs_arr, **arrays)
    info = {"run": snap["run_name"], "split": args.split, "n": len(ids), "checkpoint_epoch": epoch,
            "random_weights": args.random_weights, "max_samples": args.max_samples,
            "maps": {key: list(a.shape[1:]) for key, a in arrays.items()},
            "gradcam_layers": {name: type(layer.module).__name__ for name, layer in layers.items()},
            "gradcam_combined": combine, "attention_methods": attention_methods(model),
            "xai_config": vars(xcfg)}
    (out_dir / "explain.json").write_text(json.dumps(info, indent=2))
    log.info("%d images, maps %s -> %s", len(ids), info["maps"], out_dir / "maps.npz")

    if args.no_figures:
        return
    preds = probs_arr.argmax(axis=1)
    classes = data_cfg.classes
    masked = build_dataset(data_cfg, args.split, build_eval_transform(data_cfg.image_size), return_mask=True,
                           max_samples=args.max_samples, seed=snap["train"]["seed"])
    sweep = np.asarray(xcfg.sweep)
    picks = figure_indices(labels_arr, xcfg.figures_per_class, xcfg.figure_seed)
    for idx in picks:
        item = masked[idx]
        assert item["image_id"] == ids[idx], "dataset order changed between passes"
        mask = item["mask"].numpy().astype(bool)
        band = boundary_band(mask, xcfg.boundary_half_width)
        shown = {key.split("@")[0]: upsample(a[idx:idx + 1], data_cfg.image_size)[0]
                 for key, a in arrays.items() if not key.endswith("@true")}
        scores = {name: score_map(m, mask, xcfg.threshold, sweep, band) for name, m in shown.items()}
        verdict = "correct" if preds[idx] == labels_arr[idx] else "WRONG"
        title = (f"{snap['run_name']} | {ids[idx]} | true {classes[labels_arr[idx]]}, pred {classes[preds[idx]]} "
                 f"({probs_arr[idx, preds[idx]]:.2f}, {verdict}) | maps for the predicted class")
        save_strip(out_dir / "figures" / f"{classes[labels_arr[idx]]}_{ids[idx]}.png", denormalize(item["image"]),
                   mask, shown, scores, xcfg.threshold, title)
    log.info("%d figures -> %s", len(picks), out_dir / "figures")


if __name__ == "__main__":
    main()
