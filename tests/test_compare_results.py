"""scripts/compare_results.py: the frozen decision rule, the Holm bound, and a table built from synthetic files."""
import importlib.util
import json

import pytest

from skin_lesion_xai.config import PROJECT_DIR

PRIMARY = ["deletion_vs_random", "insertion_vs_random", "rand_full_ssim", "stab_spearman_mean", "iou", "iou_gap"]
BETTER = {"deletion_vs_random": "lower", "insertion_vs_random": "higher", "rand_full_ssim": "lower",
          "stab_spearman_mean": "higher", "iou": "higher", "iou_gap": "higher"}


def load():
    spec = importlib.util.spec_from_file_location("compare_results", PROJECT_DIR / "scripts" / "compare_results.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def endpoint(metric, diff, p_holm):
    return {"a": 0.0, "b": diff, "diff": diff, "diff_ci": {"low": diff - 0.01, "high": diff + 0.01},
            "p_two_sided": p_holm, "p_holm": p_holm, "better": BETTER[metric]}


def results(**changes):
    """All primary endpoints not significant, then the given (diff, p_holm) per metric."""
    res = {m: endpoint(m, 0.001, 0.5) for m in PRIMARY}
    for m, (diff, p) in changes.items():
        res[m] = endpoint(m, diff, p)
    return res


def test_decision_rule_cases():
    rule = load().decision_rule
    # better on deletion (lower) and rand_full_ssim (lower), nothing worse: met
    assert rule(results(deletion_vs_random=(-0.05, 0.01), rand_full_ssim=(-0.03, 0.02)), PRIMARY) == "met"
    # better on insertion (higher) instead of deletion: met
    assert rule(results(insertion_vs_random=(0.02, 0.0), rand_full_ssim=(-0.03, 0.02)), PRIMARY) == "met"
    # faithfulness significant but in the wrong direction: not met
    assert rule(results(deletion_vs_random=(0.05, 0.01), rand_full_ssim=(-0.03, 0.02)), PRIMARY) == "not met"
    # no randomisation gain: not met
    assert rule(results(deletion_vs_random=(-0.05, 0.01), rand_full_ssim=(-0.03, 0.2)), PRIMARY) == "not met"
    # significantly worse on another primary endpoint (stability lower): not met
    assert rule(results(deletion_vs_random=(-0.05, 0.01), rand_full_ssim=(-0.03, 0.02),
                        stab_spearman_mean=(-0.04, 0.001)), PRIMARY) == "not met"
    # a significant gain elsewhere does not block it
    assert rule(results(deletion_vs_random=(-0.05, 0.01), rand_full_ssim=(-0.03, 0.02), iou=(0.1, 0.0)),
                PRIMARY) == "met"
    # NaN Holm p (no valid resample) counts as not significant
    assert rule(results(deletion_vs_random=(-0.05, 0.01), rand_full_ssim=(-0.03, 0.02),
                        iou_gap=(-0.2, float("nan"))), PRIMARY) == "met"
    # p exactly at alpha is not significant
    assert rule(results(deletion_vs_random=(-0.05, 0.05), rand_full_ssim=(-0.03, 0.02)), PRIMARY) == "not met"
    res = results()
    del res["iou"]
    assert rule(res, PRIMARY) == "incomplete"


def test_holm_bound_and_p_text():
    mod = load()
    cmp = {"results": {"m1": results(), "m2": results()}}
    assert mod.holm_bound(cmp) == pytest.approx(12 * 0.002)
    assert mod.p_txt(0.0, mod.holm_bound(cmp)) == "< 0.024"
    assert mod.p_txt(0.0) == "< 0.002" and mod.p_txt(0.148) == "0.148" and mod.p_txt(float("nan")) == "—"


def test_compact_rows_from_files(tmp_path, monkeypatch):
    mod = load()
    for split, diff in [("val", -0.05), ("test", -0.01)]:
        cmp = {"n": 10, "primary": PRIMARY,
               "results": {"gradcam_fused": results(deletion_vs_random=(diff, 0.01 if split == "val" else 0.3),
                                                    rand_full_ssim=(-0.03, 0.02))}}
        (tmp_path / f"x_{split}.json").write_text(json.dumps(cmp))
    monkeypatch.setattr(mod, "CMP_DIR", tmp_path)
    monkeypatch.setattr(mod, "PROJECT_DIR", tmp_path)
    rows, src = mod.compact_rows([("cmp", "x_{split}", True)], {}, {})
    assert len(rows) == 1 and len(src) == 2
    row = rows[0]
    assert row["deletion_vs_random"] == "▲−0.050 / −0.010"
    assert row["Rule val / test"] == "met / not met"
    rows, _ = mod.compact_rows([("missing", "nothing_{split}", True)], {}, {})
    assert rows[0]["Map (B)"] == "not run"
