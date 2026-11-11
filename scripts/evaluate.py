"""Evaluate a trained run on the test split (run once per final model).

Usage: python scripts/evaluate.py --run outputs/runs/<run-name> [--n-boot 1000]
       python scripts/evaluate.py --run outputs/runs/<run-name> --val-only

Loads best.pt strictly, predicts val and test, and writes to <run>/eval/:
  metrics.json      test metrics + bootstrap 95% CIs (+ val metrics for reference)
  predictions.csv   per-image test probabilities
  confusion_matrix.png, roc.png, pr.png, history.png
Binary tasks are also reported at the thresholds fixed on val beforehand (<run>/eval_val/threshold.json from
scripts/select_threshold.py: Youden's J and a minimum sensitivity); without that file a binary run stops before
loading anything.

--val-only predicts val only (the test split is never loaded) and writes to <run>/eval_val/:
  metrics.json, predictions_val.csv, confusion_matrix.png
--data <yaml> reads the images from that config's locations instead of the paths saved with the run
(a run trained on Kaggle evaluated here); it must describe the same dataset and split.
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader

from skin_lesion_xai.config import PROJECT_DIR, DataConfig, ModelConfig, relocate_data_config
from skin_lesion_xai.data.datasets import build_dataset
from skin_lesion_xai.data.transforms import build_eval_transform
from skin_lesion_xai.eval.metrics import bootstrap_ci, compute_metrics
from skin_lesion_xai.eval.plots import plot_confusion_matrix, plot_history, plot_pr_curves, plot_roc_curves
from skin_lesion_xai.models.factory import build_model
from skin_lesion_xai.train.trainer import predict
from skin_lesion_xai.utils.checkpoint import load_checkpoint
from skin_lesion_xai.utils.device import get_device
from skin_lesion_xai.utils.logging import get_logger


def save_predictions(split_preds: dict, classes: list[str], path: Path) -> None:
    preds = pd.DataFrame(split_preds["probs"], columns=[f"p_{c}" for c in classes])
    preds.insert(0, "label", split_preds["labels"])
    preds.insert(0, "image_id", split_preds["image_ids"])
    preds.to_csv(path, index=False)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=0, help="0 = load in the main process (low-RAM machines)")
    p.add_argument("--val-only", action="store_true",
                   help="score the val split only (test is not loaded); writes to <run>/eval_val/")
    p.add_argument("--data", default=None,
                   help="data YAML giving this machine's file locations (e.g. for a run trained on Kaggle)")
    args = p.parse_args()

    run_dir = Path(args.run)
    if not run_dir.is_absolute():
        run_dir = PROJECT_DIR / run_dir
    snap = json.loads((run_dir / "config.json").read_text())
    data_cfg = relocate_data_config(snap["data"], args.data) if args.data else DataConfig(**snap["data"])
    model_cfg = ModelConfig(**snap["model"])
    out_dir = run_dir / ("eval_val" if args.val_only else "eval")
    threshold_path = run_dir / "eval_val" / "threshold.json"
    fixed = None
    if data_cfg.task == "binary" and not args.val_only:
        if not threshold_path.exists():  # checked before anything is loaded or written
            raise SystemExit(f"{threshold_path} not found: fix the thresholds on val first "
                             f"(evaluate.py --val-only, then select_threshold.py)")
        fixed = json.loads(threshold_path.read_text())
    log = get_logger("evaluate", out_dir / "evaluate.log")
    device = get_device()

    model = build_model(model_cfg, data_cfg.num_classes, pretrained=False)
    meta = load_checkpoint(run_dir / "best.pt", model)
    model.to(device)
    log.info("loaded %s (epoch %s, val %s %.4f)", run_dir / "best.pt", meta["epoch"], meta["monitor"], meta["score"])
    if fixed is not None and fixed["checkpoint_epoch"] != meta["epoch"]:
        raise SystemExit(f"{threshold_path} was chosen for epoch {fixed['checkpoint_epoch']}, best.pt is epoch "
                         f"{meta['epoch']}")

    def run_split(split: str, max_samples: int | None) -> dict:
        ds = build_dataset(data_cfg, split, build_eval_transform(data_cfg.image_size),
                           max_samples=max_samples, seed=snap["train"]["seed"])
        loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
        return predict(model, loader, device, amp=snap["train"]["amp"])

    classes = data_cfg.classes
    val = run_split("val", snap["max_val"])

    if args.val_only:
        results = {"run": snap["run_name"], "smoke": snap["smoke"], "checkpoint_epoch": meta["epoch"],
                   "val": compute_metrics(val["labels"], val["probs"], classes)}
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "metrics.json").write_text(json.dumps(results, indent=2))
        save_predictions(val, classes, out_dir / "predictions_val.csv")
        plot_confusion_matrix(results["val"]["confusion_matrix"], classes, out_dir / "confusion_matrix.png",
                              f"{snap['run_name']} (val)")
        v = results["val"]
        sens = "  ".join(f"{c} {v['per_class'][c]['sensitivity']:.3f}" for c in classes)
        log.info("val n=%d  acc %.4f  bal_acc %.4f  macro_f1 %.4f  auc %.4f  mcc %.4f -> %s",
                 v["n"], v["accuracy"], v["balanced_accuracy"], v["macro_f1"], v["auc"], v["mcc"], out_dir)
        log.info("val sensitivity: %s", sens)
        return

    # smoke runs evaluate on a subset of test as well, so they stay quick
    test = run_split("test", snap["max_val"])

    results = {"run": snap["run_name"], "smoke": snap["smoke"], "checkpoint_epoch": meta["epoch"],
               "test": compute_metrics(test["labels"], test["probs"], classes),
               "val": compute_metrics(val["labels"], val["probs"], classes)}
    results["test_ci95"] = bootstrap_ci(test["labels"], test["probs"], classes, n_boot=args.n_boot)
    if fixed is not None:  # thresholds fixed on val beforehand (select_threshold.py), never chosen here
        results["val_threshold_file"] = {"path": "eval_val/threshold.json", "created": fixed["created"],
                                         "sha256": hashlib.sha256(threshold_path.read_bytes()).hexdigest()}
        results["test_at_val_thresholds"] = {
            name: {"threshold": t["threshold"],
                   "metrics": compute_metrics(test["labels"], test["probs"], classes, threshold=t["threshold"]),
                   "ci95": bootstrap_ci(test["labels"], test["probs"], classes, threshold=t["threshold"],
                                        n_boot=args.n_boot)}
            for name, t in fixed["thresholds"].items()}

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    save_predictions(test, classes, out_dir / "predictions.csv")

    title = f"{snap['run_name']} (test)"
    plot_confusion_matrix(results["test"]["confusion_matrix"], classes, out_dir / "confusion_matrix.png", title)
    plot_roc_curves(test["labels"], test["probs"], classes, out_dir / "roc.png", title)
    plot_pr_curves(test["labels"], test["probs"], classes, out_dir / "pr.png", title)
    plot_history(run_dir / "history.csv", out_dir / "history.png", meta["monitor"])

    t = results["test"]
    log.info("test n=%d  acc %.4f  bal_acc %.4f  macro_f1 %.4f  auc %.4f  mcc %.4f -> %s",
             t["n"], t["accuracy"], t["balanced_accuracy"], t["macro_f1"], t["auc"], t["mcc"], out_dir)


if __name__ == "__main__":
    main()
