"""Paired bootstrap of explanation-trust metrics: maps of run B minus maps of run A on the same images.

Usage: python scripts/compare_xai_paired.py --runs outputs/runs/<A> outputs/runs/<B> --split test \
           --map-a gradcam --maps-b gradcam_local gradcam_fused [--method-a gradcam] [--method-b gradcam] [--smoke]

Reads <run>/trust_<split>[_<method>][_smoke]/per_image.csv (scripts/eval_trust.py). Localisation scores come
from that file when eval_trust.py wrote them (`localisation: true`), otherwise from
<run>/xai_<split>[_smoke]/per_image.csv (scripts/eval_xai.py, predicted-class maps; skipped with a log line if
missing). A and B may be the same run with two methods (does a method change the trust of one model's maps?).
--map-a takes one map (compared with every map of B) or one map per map of B (compared pairwise).
Both sides must list the same images with the same labels: if one side covers a subset of the other's images
(a Score-CAM subset run against a full run), the larger side is restricted to the subset, in its order.
For every metric: values of A and B, B - A, percentile 95% CIs from the same resampled image sets (trust
config n_boot, seed), and a two-sided bootstrap p-value. The endpoints in `primary` (for every map of B) are
Holm-corrected together; everything else is exploratory.
  *_vs_random   the map's deletion / insertion AUC minus that of the run's own random reference map
  iou_gap       mean IoU on correctly predicted images minus on wrong ones (each model its own correctness)
  rand_full_abs_{spearman,hog}  |rank correlation| of the trained and fully randomised maps: 0 = nothing of the
                trained map is left; the signed value would reward anti-correlation (changed after the val run)
  stab_spearman_no_noise  stability Spearman over the flip and brightness perturbations only (unchanged
                predictions), because SmoothGrad averages over input noise by design (Part B)
Writes outputs/comparisons/<dataset>_xai_<split>_<B>[-<method>]_vs_<A>[-<method>].json (+ .log).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from skin_lesion_xai.config import PROJECT_DIR, TrustConfig
from skin_lesion_xai.eval.metrics import holm, paired_bootstrap_per_image
from skin_lesion_xai.utils.logging import get_logger
from skin_lesion_xai.xai import trust_dir, xai_dir

# metric -> which direction means a more trustworthy map
DIRECTION = {
    "deletion_auc": "lower", "insertion_auc": "higher", "deletion_vs_random": "lower", "insertion_vs_random": "higher",
    "rand_full_abs_spearman": "lower", "rand_full_ssim": "lower", "rand_full_abs_hog": "lower",
    "stab_spearman_mean": "higher", "stab_ssim_mean": "higher", "stab_spearman_no_noise": "higher",
    "iou": "higher", "energy_in_mask": "higher", "pointing": "higher", "iou_gap": "higher",
}
LOCALISATION = ("iou", "energy_in_mask", "pointing")


def load(run_dir: Path, split: str, smoke: bool, method: str) -> tuple[pd.DataFrame, pd.DataFrame | None, Path]:
    trust = pd.read_csv(trust_dir(run_dir, split, smoke, method) / "per_image.csv")
    if all(c in trust.columns for c in LOCALISATION):
        return trust, None, trust_dir(run_dir, split, smoke, method)
    xai_path = xai_dir(run_dir, split, smoke=smoke) / "per_image.csv"
    if method != "gradcam" or not xai_path.exists():  # eval_xai.py scores Grad-CAM maps only
        return trust, None, xai_path
    xai = pd.read_csv(xai_path)
    return trust, xai[xai["target"] == "pred"], xai_path


def has_localisation(trust: pd.DataFrame, xai: pd.DataFrame | None) -> bool:
    return xai is not None or all(c in trust.columns for c in LOCALISATION)


def table(trust: pd.DataFrame, xai: pd.DataFrame | None, map_name: str) -> pd.DataFrame:
    """One row per image for one map: trust metrics, *_vs_random, localisation."""
    t = trust[trust["map"] == map_name].reset_index(drop=True)
    if t.empty:
        raise SystemExit(f"no rows for map {map_name}")
    ref = trust[trust["map"] == "ref_random"].set_index("image_id").loc[t["image_id"]].reset_index()
    for mode in ("deletion", "insertion"):
        t[f"{mode}_vs_random"] = t[f"{mode}_auc"].to_numpy() - ref[f"{mode}_auc"].to_numpy()
    for k in ("spearman", "hog"):
        if f"rand_full_{k}" in t.columns:
            t[f"rand_full_abs_{k}"] = t[f"rand_full_{k}"].abs()
    quiet = [c[len("stab_"):-len("_spearman")] for c in t.columns
             if c.startswith("stab_") and c.endswith("_spearman") and not c.startswith(("stab_noise", "stab_spearman"))]
    if quiet:
        values = t[[f"stab_{n}_spearman" for n in quiet]].astype(float)
        changed = t[[f"stab_{n}_pred_changed" for n in quiet]].fillna(False).astype(bool).to_numpy()
        t["stab_spearman_no_noise"] = values.where(~changed).mean(axis=1)
    if xai is not None:
        x = xai[xai["method"] == map_name].set_index("image_id")
        if set(x.index) != set(t["image_id"]):
            raise SystemExit(f"localisation rows of {map_name} cover other images than the trust rows")
        for col in LOCALISATION:
            t[col] = x.loc[t["image_id"], col].to_numpy()
    return t


def align(a: pd.DataFrame, b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Same images in the same order; a side covering more images is restricted to the other's (its order)."""
    ids_a, ids_b = set(a["image_id"]), set(b["image_id"])
    if ids_a != ids_b:
        if ids_b < ids_a:
            a = a.set_index("image_id").loc[b["image_id"]].reset_index()
        elif ids_a < ids_b:
            b = b.set_index("image_id").loc[a["image_id"]].reset_index()
        else:
            raise SystemExit("the two runs cover different images, and neither is a subset of the other")
    if not (a["image_id"].equals(b["image_id"]) and a["label"].equals(b["label"])):
        raise SystemExit("the two runs do not list the same images and labels in the same order")
    return a, b, len(ids_a ^ ids_b)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs=2, required=True, metavar=("A", "B"))
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--map-a", nargs="+", default=["gradcam"])
    p.add_argument("--maps-b", nargs="+", default=["gradcam_local", "gradcam_fused"])
    p.add_argument("--method-a", default="gradcam")
    p.add_argument("--method-b", default="gradcam")
    p.add_argument("--trust-config", default="configs/xai/trust.yaml")
    p.add_argument("--smoke", action="store_true", help="compare the eval_trust.py --max-samples outputs")
    args = p.parse_args()
    if len(args.map_a) not in (1, len(args.maps_b)):
        raise SystemExit("--map-a takes one map, or one per map of --maps-b")
    maps_a = args.map_a * len(args.maps_b) if len(args.map_a) == 1 else args.map_a

    tcfg = TrustConfig.from_yaml(args.trust_config)
    dirs = [Path(r) if Path(r).is_absolute() else PROJECT_DIR / r for r in args.runs]
    tag = lambda d, m: d.name + ("" if m == "gradcam" else f"-{m}")
    stem = (f"{dirs[0].name.split('_')[0]}_xai_{args.split}{'_smoke' if args.smoke else ''}_"
            f"{tag(dirs[1], args.method_b)}_vs_{tag(dirs[0], args.method_a)}")
    out_dir = PROJECT_DIR / "outputs" / "comparisons"
    (trust_a, xai_a, path_a), (trust_b, xai_b, path_b) = (
        load(d, args.split, args.smoke, m) for d, m in zip(dirs, (args.method_a, args.method_b)))
    with_loc = has_localisation(trust_a, xai_a), has_localisation(trust_b, xai_b)
    if with_loc[0] != with_loc[1]:
        raise SystemExit(f"localisation scores exist for only one side ({path_a}, {path_b})")
    pairs = {}  # B map -> (A table, B table), checked before anything is written
    for map_a, map_b in zip(maps_a, args.maps_b):
        a, b, dropped = align(table(trust_a, xai_a, map_a), table(trust_b, xai_b, map_b))
        pairs[map_b] = (map_a, a, b)
    log = get_logger("compare_xai_paired", out_dir / f"{stem}.log")
    if not with_loc[0]:
        log.info("no localisation scores (%s): localisation metrics skipped", path_a)
    if dropped:
        log.info("one side covers %d more images; it is restricted to the other side's subset", dropped)

    results, primary_p = {}, {}
    for map_b, (map_a, a, b) in pairs.items():
        results[map_b] = {}
        metrics = [m for m in DIRECTION if m in a.columns and m in b.columns or m == "iou_gap" and "iou" in a.columns]
        for metric in metrics:
            if metric == "iou_gap":
                r = paired_bootstrap_per_image(a["iou"], b["iou"], tcfg.n_boot, seed=tcfg.seed,
                                               correct_a=a["correct"], correct_b=b["correct"])
            else:
                r = paired_bootstrap_per_image(a[metric], b[metric], tcfg.n_boot, seed=tcfg.seed)
            r["better"] = DIRECTION[metric]
            results[map_b][metric] = r
            if metric in tcfg.primary and not np.isnan(r["p_two_sided"]):  # NaN: no valid resample, no test
                primary_p[f"{map_b}:{metric}"] = r["p_two_sided"]
    missing = [m for m in tcfg.primary if not all(m in res and res[m]["n_valid"] > 0 for res in results.values())]
    for key, adj in holm(primary_p).items():
        map_b, metric = key.split(":")
        results[map_b][metric]["p_holm"] = adj

    n = len(next(iter(pairs.values()))[1])
    out = {"a": [f"{dirs[0].name}:{args.method_a}:{m}" for m in maps_a],
           "b": [f"{dirs[1].name}:{args.method_b}:{m}" for m in args.maps_b],
           "split": args.split, "smoke": args.smoke, "n": int(n), "n_boot": tcfg.n_boot, "seed": tcfg.seed,
           "trust_config": args.trust_config, "primary": tcfg.primary, "primary_missing": missing,
           "results": results}
    (out_dir / f"{stem}.json").write_text(json.dumps(out, indent=2))

    log.info("%s, n=%d, %d paired resamples; B - A, better = direction of a more trustworthy map", args.split,
             n, tcfg.n_boot)
    for map_b, res in results.items():
        log.info("B = %s:%s:%s  vs  A = %s:%s:%s", dirs[1].name, args.method_b, map_b, dirs[0].name,
                 args.method_a, pairs[map_b][0])
        for metric, r in res.items():
            holm_txt = f"  Holm {r['p_holm']:.3f}" if "p_holm" in r else ""
            log.info("  %-22s (%-6s) A %.4f  B %.4f  B-A %+.4f [%+.4f, %+.4f]  p %.3f%s", metric, r["better"],
                     r["a"], r["b"], r["diff"], r["diff_ci"]["low"], r["diff_ci"]["high"], r["p_two_sided"], holm_txt)
    if missing:
        log.info("primary endpoints not available here: %s", missing)
    log.info("-> %s", out_dir / f"{stem}.json")


if __name__ == "__main__":
    main()
