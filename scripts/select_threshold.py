"""Choose the binary decision thresholds on the validation split and fix them before any test run.

Usage: python scripts/select_threshold.py --run outputs/runs/<run-name> [--min-sensitivity 0.80]

Reads only <run>/eval_val/predictions_val.csv (from evaluate.py --val-only); no model or image is loaded.
Two cutoffs on P(melanoma), predicting melanoma when P >= cutoff:
  youden            maximises sensitivity + specificity - 1 (Youden's J)
  sensitivity_0.80  highest cutoff with val sensitivity >= --min-sensitivity (best specificity among those)
Writes <run>/eval_val/threshold.json: both cutoffs, val metrics at each, and the SHA-256 of the predictions file.
evaluate.py (test) reads this file and fails without it. This script refuses to run once <run>/eval/ (the test
output) exists, so a threshold can never be picked after the test set has been seen.

Val metrics at a val-chosen cutoff are optimistic (chosen and measured on the same images); only the test
metrics at the fixed cutoff are an unbiased estimate.
"""
import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from skin_lesion_xai.config import PROJECT_DIR
from skin_lesion_xai.eval.metrics import compute_metrics, sensitivity_threshold, youden_threshold
from skin_lesion_xai.utils.logging import get_logger


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--min-sensitivity", type=float, default=0.80)
    args = p.parse_args()

    run_dir = Path(args.run)
    if not run_dir.is_absolute():
        run_dir = PROJECT_DIR / run_dir
    if (run_dir / "eval").exists():
        raise SystemExit(f"{run_dir / 'eval'} exists: the test set has been evaluated, a threshold can no longer be chosen")
    val_dir = run_dir / "eval_val"
    pred_path = val_dir / "predictions_val.csv"
    if not pred_path.exists():
        raise SystemExit(f"{pred_path} not found: run evaluate.py --run {args.run} --val-only first")
    val_metrics = json.loads((val_dir / "metrics.json").read_text())
    log = get_logger("select_threshold", val_dir / "select_threshold.log")

    preds = pd.read_csv(pred_path)
    prob_cols = [c for c in preds.columns if c.startswith("p_")]
    if len(prob_cols) != 2:
        raise SystemExit(f"threshold selection is for binary tasks; {pred_path} has {len(prob_cols)} classes")
    classes = [c[2:] for c in prob_cols]
    y, probs = preds["label"].to_numpy(), preds[prob_cols].to_numpy()

    sens_key = f"sensitivity_{args.min_sensitivity:.2f}"
    cutoffs = {"youden": youden_threshold(y, probs[:, 1]),
               sens_key: sensitivity_threshold(y, probs[:, 1], args.min_sensitivity)}
    out = {"run": val_metrics["run"], "checkpoint_epoch": val_metrics["checkpoint_epoch"],
           "source": "eval_val/predictions_val.csv",
           "source_sha256": hashlib.sha256(pred_path.read_bytes()).hexdigest(),
           "created": datetime.now().isoformat(timespec="seconds"),
           "rule": f"predict {classes[1]} when p_{classes[1]} >= threshold",
           "min_sensitivity": args.min_sensitivity, "thresholds": {}}
    for name, thr in cutoffs.items():
        m = compute_metrics(y, probs, classes, threshold=thr)
        out["thresholds"][name] = {"threshold": thr, "n_predicted_positive": int((probs[:, 1] >= thr).sum()),
                                   "val": m}
    (val_dir / "threshold.json").write_text(json.dumps(out, indent=2))

    log.info("%s, val n=%d (%d %s), checkpoint epoch %s", out["run"], len(y), int(y.sum()), classes[1],
             out["checkpoint_epoch"])
    log.info("%-18s %9s %8s %6s %6s %7s %6s %6s", "cutoff", "threshold", "called+", "sens", "spec", "bal_acc",
             "F1+", "MCC")
    for name, t in out["thresholds"].items():
        m = t["val"]
        log.info("%-18s %9.4f %8d %6.3f %6.3f %7.3f %6.3f %6.3f", name, t["threshold"], t["n_predicted_positive"],
                 m["sensitivity"], m["specificity"], m["balanced_accuracy"], m["per_class"][classes[1]]["f1"],
                 m["mcc"])
    log.info("-> %s (val metrics at a val-chosen cutoff are optimistic)", val_dir / "threshold.json")


if __name__ == "__main__":
    main()
