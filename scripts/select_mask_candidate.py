"""Pick a mask-guided fine-tuning candidate by the rule fixed before training (Part B, EXPLORATORY; val only).

Usage: python scripts/select_mask_candidate.py --stage <name> --candidates outputs/runs/<run> ... \
           [--source outputs/runs/ham10000_hybrid] [--max-f1-drop 0.01]

Rule (docs/report_notes.md, "Part B"):
  eligible  best val macro-F1 (train_summary.json) >= the source run's best val macro-F1 - max_f1_drop
  choose    the highest val insertion_vs_random of the fused Grad-CAM on the screening subset
            (trust_val_screen300/: insertion AUC minus that of the run's random map, image mean), rounded to
            3 decimals; ties broken by the higher stab_spearman_mean of the fused map
The source run is screened too and listed for reference, but it is not a candidate. Refuses if any run has
been evaluated on test (eval/ or a trust_test* folder). Writes outputs/comparisons/ham10000_mask_guided_<stage>.json.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from skin_lesion_xai.config import PROJECT_DIR

MAP = "gradcam_fused"
SCREEN = "trust_val_screen300"


def screen(run_dir: Path) -> dict:
    summary = json.loads((run_dir / "train_summary.json").read_text())
    frame = pd.read_csv(run_dir / SCREEN / "per_image.csv")
    fused = frame[frame["map"] == MAP].set_index("image_id")
    rand = frame[frame["map"] == "ref_random"].set_index("image_id").loc[fused.index]
    return {"run": run_dir.name, "val_macro_f1": summary["best_val_score"], "best_epoch": summary["best_epoch"],
            "n_screen": int(len(fused)),
            "insertion_vs_random": float((fused["insertion_auc"] - rand["insertion_auc"]).mean()),
            "deletion_vs_random": float((fused["deletion_auc"] - rand["deletion_auc"]).mean()),
            "stab_spearman_mean": float(fused["stab_spearman_mean"].mean()),
            "rand_full_ssim": float(fused["rand_full_ssim"].mean()), "iou": float(fused["iou"].mean())}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True, help="name of this selection step, e.g. lambda or consistency")
    p.add_argument("--candidates", nargs="+", required=True)
    p.add_argument("--source", default="outputs/runs/ham10000_hybrid")
    p.add_argument("--max-f1-drop", type=float, default=0.01)
    args = p.parse_args()

    resolve = lambda r: Path(r) if Path(r).is_absolute() else PROJECT_DIR / r
    source, candidates = resolve(args.source), [resolve(r) for r in args.candidates]
    for run in candidates:
        if (run / "eval").exists() or any(run.glob("trust_test*")):
            raise SystemExit(f"{run.name} has been evaluated on test; selection must happen before that")
    ref = screen(source)
    rows = [screen(r) for r in candidates]
    floor = ref["val_macro_f1"] - args.max_f1_drop
    for r in rows:
        r["eligible"] = r["val_macro_f1"] >= floor
    eligible = [r for r in rows if r["eligible"]]
    chosen = max(eligible, key=lambda r: (round(r["insertion_vs_random"], 3), r["stab_spearman_mean"]))["run"] \
        if eligible else None
    out = {"stage": args.stage, "rule": f"val macro-F1 >= {floor:.4f}, then max insertion_vs_random ({MAP}, "
                                         f"3 decimals), ties by stab_spearman_mean", "source": ref,
           "candidates": rows, "chosen": chosen}
    path = PROJECT_DIR / "outputs" / "comparisons" / f"ham10000_mask_guided_{args.stage}.json"
    path.write_text(json.dumps(out, indent=2))
    print(pd.DataFrame([ref | {"eligible": "source"}] + rows).to_string(index=False))
    print(f"chosen: {chosen}  -> {path}")


if __name__ == "__main__":
    main()
