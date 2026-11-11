"""Grad-CAM of the figure images before and after full model-parameter randomisation (for the report figure).

Usage: python scripts/explain_randomised.py --run outputs/runs/<run-name> [--split test]

The images are the figure images of scripts/explain.py (figures_per_class per class, figure_seed in
configs/xai/default.yaml; the same images in every run). The maps are the trust maps of the run's model in
configs/xai/trust.yaml, for the class the trained model predicts. Randomisation is the end state of the M4
cascade: every block of xai/randomization.py re-initialised at once, after seeding with the trust seed. It is
an illustration of the parameter-randomisation check, not a rerun of it: the random weights are not the ones
drawn in eval_trust.py, whose scores stay the reported ones.
Writes <run>/xai_<split>_randomised_examples/: maps.npz (image_ids, labels, preds, "<map>@trained" and
"<map>@randomised" raw maps) and info.json (settings and per-image Spearman / SSIM of each pair).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from skin_lesion_xai.config import PROJECT_DIR, DataConfig, ModelConfig, TrustConfig, XaiConfig
from skin_lesion_xai.data.datasets import build_dataset
from skin_lesion_xai.data.transforms import build_eval_transform
from skin_lesion_xai.models.factory import build_model
from skin_lesion_xai.utils.checkpoint import load_checkpoint
from skin_lesion_xai.utils.device import get_device
from skin_lesion_xai.utils.logging import get_logger
from skin_lesion_xai.utils.seed import seed_everything
from skin_lesion_xai.xai.cam_variants import GradCAMExplainer
from skin_lesion_xai.xai.gradcam import freeze
from skin_lesion_xai.xai.localization import upsample
from skin_lesion_xai.xai.randomization import cascade, randomize_step
from skin_lesion_xai.xai.similarity import similarities
from skin_lesion_xai.xai.visualize import figure_indices


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--xai-config", default="configs/xai/default.yaml")
    p.add_argument("--trust-config", default="configs/xai/trust.yaml")
    args = p.parse_args()

    run_dir = Path(args.run)
    if not run_dir.is_absolute():
        run_dir = PROJECT_DIR / run_dir
    snap = json.loads((run_dir / "config.json").read_text())
    data_cfg, model_cfg = DataConfig(**snap["data"]), ModelConfig(**snap["model"])
    xcfg, tcfg = XaiConfig.from_yaml(args.xai_config), TrustConfig.from_yaml(args.trust_config)
    map_names = tcfg.maps[model_cfg.name]
    out_dir = run_dir / f"xai_{args.split}_randomised_examples"
    log = get_logger("explain_randomised", out_dir / "explain_randomised.log")
    device = get_device()
    seed_everything(snap["train"]["seed"])

    model = build_model(model_cfg, data_cfg.num_classes, pretrained=False)
    epoch = load_checkpoint(run_dir / "best.pt", model)["epoch"]
    model = freeze(model).to(device).eval()
    explain = GradCAMExplainer(model)

    ds = build_dataset(data_cfg, args.split, build_eval_transform(data_cfg.image_size))
    picks = figure_indices(ds.labels, xcfg.figures_per_class, xcfg.figure_seed)
    items = [ds[i] for i in picks]
    x = torch.stack([it["image"] for it in items]).to(device)

    logits, cams = explain(x, {"pred": None})
    preds = logits.argmax(dim=1)
    trained = {m: cams["pred"][m] for m in map_names}
    seed_everything(tcfg.seed)
    randomize_step(model, [path for _, paths in cascade(model) for path in paths])
    _, cams = explain(x, {"pred": preds})
    randomised = {m: cams["pred"][m] for m in map_names}

    size = data_cfg.image_size
    sims = {m: [similarities(a, b, tcfg.hog_pixels_per_cell, with_hog=False)
                for a, b in zip(upsample(trained[m], size), upsample(randomised[m], size))] for m in map_names}
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "maps.npz", image_ids=np.array([it["image_id"] for it in items]),
                        labels=np.array([int(it["label"]) for it in items]), preds=preds.cpu().numpy(),
                        **{f"{m}@trained": trained[m].astype(np.float32) for m in map_names},
                        **{f"{m}@randomised": randomised[m].astype(np.float32) for m in map_names})
    info = {"run": snap["run_name"], "split": args.split, "checkpoint_epoch": epoch, "maps": map_names,
            "positions": picks, "randomisation": "full (all cascade blocks)", "seed": tcfg.seed,
            "similarity": {m: [{k: float(v) for k, v in s.items()} for s in sims[m]] for m in map_names}}
    (out_dir / "info.json").write_text(json.dumps(info, indent=2))
    for m in map_names:
        log.info("%s: mean Spearman %.3f, SSIM %.3f over %d images", m, np.mean([s["spearman"] for s in sims[m]]),
                 np.mean([s["ssim"] for s in sims[m]]), len(picks))
    log.info("-> %s", out_dir)


if __name__ == "__main__":
    main()
