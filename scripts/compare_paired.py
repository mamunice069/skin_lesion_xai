"""Paired bootstrap comparison of two evaluated runs on the same test images (model B minus model A).

Usage: python scripts/compare_paired.py --runs outputs/runs/<A> outputs/runs/<B> [--n-boot 1000] [--seed 0]

Reads only <run>/eval/predictions.csv (and, for binary tasks, the cutoffs fixed beforehand in
<run>/eval_val/threshold.json); no model or image is loaded and no threshold is chosen here. Both prediction
files must list the same images with the same labels in the same order.
  binary      operating points argmax and each model's own Youden and minimum-sensitivity cutoffs: AUC, AP,
              sensitivity / specificity. Each threshold file must have the SHA-256 that evaluate.py recorded in
              <run>/eval/metrics.json (so it was not changed after the test run).
  multiclass  argmax: macro-F1, balanced accuracy, accuracy, MCC, kappa, OvR macro AUC / AP, per-class sensitivity.
Writes outputs/comparisons/<dataset>_test_<B>_vs_<A>.json (+ .log): per metric the values of A and B, B - A,
percentile 95% CIs from the shared resamples, and a two-sided bootstrap p-value.
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from skin_lesion_xai.config import PROJECT_DIR
from skin_lesion_xai.eval.metrics import paired_bootstrap, paired_bootstrap_multiclass
from skin_lesion_xai.utils.logging import get_logger


def load_run(run_dir: Path) -> tuple[pd.DataFrame, dict | None, dict]:
    """Test predictions, the fixed binary thresholds (None for a multiclass run) and eval/metrics.json."""
    metrics = json.loads((run_dir / "eval" / "metrics.json").read_text())
    pred = pd.read_csv(run_dir / "eval" / "predictions.csv")
    if sum(c.startswith("p_") for c in pred.columns) != 2:
        return pred, None, metrics
    thr_path = run_dir / "eval_val" / "threshold.json"
    recorded = metrics.get("val_threshold_file", {}).get("sha256")
    if recorded is None or hashlib.sha256(thr_path.read_bytes()).hexdigest() != recorded:
        raise SystemExit(f"{thr_path} does not match the threshold file used by the test run of {run_dir.name}")
    cutoffs = json.loads(thr_path.read_text())["thresholds"]
    thresholds = {"argmax": None} | {name: t["threshold"] for name, t in cutoffs.items()}
    return pred, thresholds, metrics


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs=2, required=True, metavar=("A", "B"))
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    dirs = [Path(r) if Path(r).is_absolute() else PROJECT_DIR / r for r in args.runs]
    (pred_a, thr_a, met_a), (pred_b, thr_b, _) = load_run(dirs[0]), load_run(dirs[1])
    if not (pred_a["image_id"].equals(pred_b["image_id"]) and pred_a["label"].equals(pred_b["label"])):
        raise SystemExit("the two runs' test predictions do not list the same images and labels in the same order")
    prob_cols = [c for c in pred_a.columns if c.startswith("p_")]
    if prob_cols != [c for c in pred_b.columns if c.startswith("p_")]:
        raise SystemExit("the two runs' predictions have different classes")
    y = pred_a["label"].to_numpy()

    dataset = met_a["run"].split("_")[0]
    out_dir = PROJECT_DIR / "outputs" / "comparisons"
    stem = f"{dataset}_test_{dirs[1].name}_vs_{dirs[0].name}"
    log = get_logger("compare_paired", out_dir / f"{stem}.log")
    out = {"a": dirs[0].name, "b": dirs[1].name, "split": "test", "n": int(len(pred_a)),
           "n_boot": args.n_boot, "seed": args.seed}
    if thr_a is None:
        classes = [c[2:] for c in prob_cols]
        res = paired_bootstrap_multiclass(y, pred_a[prob_cols].to_numpy(), pred_b[prob_cols].to_numpy(), classes,
                                          n_boot=args.n_boot, seed=args.seed)
        out |= {"task": "multiclass", "classes": classes, "metrics": res}
        counts = f"{len(classes)} classes"
    else:
        res = paired_bootstrap(y, pred_a[prob_cols].to_numpy(), pred_b[prob_cols].to_numpy(),
                               thr_a, thr_b, n_boot=args.n_boot, seed=args.seed)
        out |= {"task": "binary", "n_positive": int(y.sum()), "thresholds": {"a": thr_a, "b": thr_b}, "metrics": res}
        counts = f"{out['n_positive']} positive"
    (out_dir / f"{stem}.json").write_text(json.dumps(out, indent=2))

    log.info("test n=%d (%s), %d paired resamples (%d valid): B = %s minus A = %s", out["n"], counts,
             args.n_boot, next(iter(res.values()))["n_valid"], out["b"], out["a"])
    for k, r in res.items():
        log.info("%-30s A %.4f [%.4f, %.4f]  B %.4f [%.4f, %.4f]  B-A %+.4f [%+.4f, %+.4f]  p %.3f", k,
                 r["a"], r["a_ci"]["low"], r["a_ci"]["high"], r["b"], r["b_ci"]["low"], r["b_ci"]["high"],
                 r["diff"], r["diff_ci"]["low"], r["diff_ci"]["high"], r["p_two_sided"])
    log.info("-> %s", out_dir / f"{stem}.json")


if __name__ == "__main__":
    main()
