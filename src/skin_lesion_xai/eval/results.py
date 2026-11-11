"""Shared by scripts/compare_results.py (tables) and scripts/make_figures.py (figures): which runs and comparison
files make up the reported results, how to read them, and the frozen decision rule of M4 / Part B.

The run lists say which runs appear and in which role (mirroring docs/report_notes.md); a listed run whose files
are missing is reported as "not run". Nothing here holds a result: every number is read from the run outputs.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from skin_lesion_xai.config import PROJECT_DIR

RUNS_DIR = PROJECT_DIR / "outputs" / "runs"
CMP_DIR = PROJECT_DIR / "outputs" / "comparisons"
ALPHA = 0.05
MIN_P = 0.002  # smallest non-zero two-sided bootstrap p with 1,000 resamples
EXPLORATORY = "EXPLORATORY"

# (run, label, role) in table order
HAM_RUNS = [
    ("ham10000_resnet50", "ResNet-50", "baseline"),
    ("ham10000_convnext_t", "ConvNeXt-T", "baseline"),
    ("ham10000_vit_b16_lr3e-5", "ViT-B/16 (lr 3e-5)", "baseline"),
    ("ham10000_swin_t", "Swin-T", "baseline; global branch alone"),
    ("ham10000_effnetv2_s", "EfficientNetV2-S", "baseline (ImageNet norm)"),
    ("ham10000_effnetv2_s_pnorm", "EfficientNetV2-S, pretrained norm", "local branch alone (norm fix)"),
    ("ham10000_vit_b16_pnorm_lr3e-5", "ViT-B/16 (lr 3e-5), pretrained norm", "norm fix"),
    ("ham10000_hybrid", "Hybrid, cross-attention", "proposed"),
    ("ham10000_hybrid_concat", "Hybrid, concat", "fusion ablation"),
    ("ham10000_convnext_t_hair", "ConvNeXt-T, hair removed", "hair ablation"),
    ("ham10000_hybrid_hair", "Hybrid, hair removed", "hair ablation; best on test, not pre-specified"),
    ("ham10000_hybrid_mask_l0.1_m0", "Hybrid, mask-guided (λ 0.1, μ 0)", f"{EXPLORATORY} (Part B step 5)"),
]
ISIC_RUNS = [
    ("isic2020_convnext_t", "ConvNeXt-T"),
    ("isic2020_hybrid", "Hybrid, cross-attention"),
    ("isic2020_effnetv2_s_pnorm", "EfficientNetV2-S, pretrained norm"),
]
MASK_RUN = "ham10000_hybrid_mask_l0.1_m0"
XAI_RUNS = [r for r in HAM_RUNS if r[0] != MASK_RUN]
MAP_LABELS = {
    "gradcam": "Grad-CAM",
    "gradcam_local": "Grad-CAM, EffNetV2 branch",
    "gradcam_global": "Grad-CAM, Swin branch",
    "gradcam_fused": "Grad-CAM, fused",
    "rollout": "attention rollout",
    "xattn_local_to_global": "cross-attn, EffNetV2 → Swin",
    "xattn_global_to_local": "cross-attn, Swin → EffNetV2",
}
# paired classification files: (file stem, label, Holm family or None)
HAM_PAIRED = [
    ("ham10000_test_ham10000_hybrid_vs_ham10000_convnext_t", "Hybrid − ConvNeXt-T", "pre-specified"),
    ("ham10000_test_ham10000_hybrid_vs_ham10000_hybrid_concat", "Hybrid − concat hybrid", "pre-specified"),
    (f"ham10000_test_{MASK_RUN}_vs_ham10000_hybrid", f"Mask-guided − hybrid ({EXPLORATORY})", None),
]
ISIC_PAIRED = "isic2020_test_isic2020_hybrid_vs_isic2020_convnext_t"
# hybrid minus its local branch alone (what the Swin branch + fusion add); pending until the Kaggle run is evaluated
ISIC_PAIRED_LOCAL = "isic2020_test_isic2020_hybrid_vs_isic2020_effnetv2_s_pnorm"
NOT_WORSE_MARGIN = -0.01  # Part B step 5: mask-guided "not worse" if the macro-F1 diff CI lower bound > -0.01
M4_STEM = "ham10000_xai_{split}_ham10000_hybrid_vs_ham10000_convnext_t"
# Part B comparisons: (label, file stem with {split} (and {m}), decision rule applies)
STEP4 = [
    ("(a) hybrid vs ConvNeXt-T, both {M}", "ham10000_xai_{split}_ham10000_hybrid-{m}_vs_ham10000_convnext_t-{m}", True),
    ("(b) hybrid: {M} vs its Grad-CAM", "ham10000_xai_{split}_ham10000_hybrid-{m}_vs_ham10000_hybrid", True),
    ("(c) ConvNeXt-T: {M} vs its Grad-CAM", "ham10000_xai_{split}_ham10000_convnext_t-{m}_vs_ham10000_convnext_t", False),
]
STEP4_METHODS = [("smoothgrad", "SmoothGrad-CAM"), ("scorecam", "Score-CAM")]
STEP5 = [
    ("(d) mask-guided vs hybrid", f"ham10000_xai_{{split}}_{MASK_RUN}_vs_ham10000_hybrid", True),
    ("(e) mask-guided vs ConvNeXt-T", f"ham10000_xai_{{split}}_{MASK_RUN}_vs_ham10000_convnext_t", True),
]


def read_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def rel(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_DIR).as_posix()
    except ValueError:
        return path.as_posix()


def is_nan(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def holm_bound(cmp: dict) -> float:
    """Upper bound of a Holm-adjusted p whose raw p is 0: (number of Holm-corrected tests) x 0.002."""
    m = sum("p_holm" in r for res in cmp["results"].values() for r in res.values())
    return min(1.0, m * MIN_P)


def significant(r: dict) -> bool:
    return not is_nan(r.get("p_holm")) and r["p_holm"] < ALPHA


def direction(r: dict) -> int:
    """+1 significantly better, -1 significantly worse, 0 not significant (Holm p < 0.05)."""
    if not significant(r) or r["diff"] == 0:
        return 0
    good = r["diff"] < 0 if r["better"] == "lower" else r["diff"] > 0
    return 1 if good else -1


def decision_rule(results: dict, primary: list[str]) -> str:
    """The frozen M4 rule (docs/report_notes.md): B is more trustworthy than A only if it is significantly better
    (Holm p < 0.05) on deletion_vs_random or insertion_vs_random, and on rand_full_ssim, and significantly worse
    on no other primary endpoint. Returns "met", "not met" or "incomplete" (a primary endpoint missing)."""
    if any(m not in results for m in primary):
        return "incomplete"
    d = {m: direction(results[m]) for m in primary}
    faithful = d.get("deletion_vs_random") == 1 or d.get("insertion_vs_random") == 1
    random_ok = d.get("rand_full_ssim") == 1
    return "met" if faithful and random_ok and not any(v == -1 for v in d.values()) else "not met"


def load_comparisons(stem: str, **fmt) -> dict[str, tuple[Path, dict | None]]:
    """{split: (path, comparison or None)} for a val / test pair of comparison files."""
    out = {}
    for split in ("val", "test"):
        path = CMP_DIR / f"{stem.format(split=split, **fmt)}.json"
        out[split] = (path, read_json(path))
    return out
