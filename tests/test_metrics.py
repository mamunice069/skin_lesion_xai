"""Metrics on small inputs whose answers are worked out by hand."""
import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from skin_lesion_xai.config import PROJECT_DIR, DataConfig, ModelConfig
from skin_lesion_xai.eval.metrics import (
    bootstrap_ci, compute_metrics, paired_bootstrap, paired_bootstrap_multiclass, sensitivity_threshold,
    youden_threshold,
)
from skin_lesion_xai.train.optim import warmup_cosine_factor

BINARY = ["benign", "melanoma"]


def test_binary_by_hand():
    # y:    0    0    0    0    1    1
    # pred: 0    0    0    1    1    0      -> TP=1 FN=1 TN=3 FP=1
    y = np.array([0, 0, 0, 0, 1, 1])
    p1 = np.array([0.1, 0.2, 0.3, 0.6, 0.9, 0.4])
    m = compute_metrics(y, np.stack([1 - p1, p1], axis=1), BINARY)
    assert m["accuracy"] == pytest.approx(4 / 6)
    assert m["sensitivity"] == pytest.approx(1 / 2)  # TP / (TP + FN)
    assert m["specificity"] == pytest.approx(3 / 4)  # TN / (TN + FP)
    # AUC: positives {0.9, 0.4} vs negatives {0.1, 0.2, 0.3, 0.6}: 0.9 beats 4, 0.4 beats 3 -> 7/8
    assert m["auc"] == pytest.approx(7 / 8)
    assert m["confusion_matrix"] == [[3, 1], [1, 1]]


def test_threshold_changes_predictions():
    y = np.array([0, 0, 1, 1])
    p1 = np.array([0.1, 0.3, 0.35, 0.8])
    probs = np.stack([1 - p1, p1], axis=1)
    assert compute_metrics(y, probs, BINARY)["sensitivity"] == pytest.approx(0.5)
    assert compute_metrics(y, probs, BINARY, threshold=0.33)["sensitivity"] == pytest.approx(1.0)
    assert youden_threshold(y, p1) == pytest.approx(0.35)


def test_sensitivity_threshold_is_the_highest_cutoff_reaching_the_target():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])
    p1 = np.array([0.1, 0.2, 0.5, 0.7, 0.3, 0.4, 0.6, 0.8, 0.9])
    # positives sorted: 0.9 0.8 0.6 0.4 0.3 -> sensitivity >= 0.8 needs 4 of 5, i.e. cutoff 0.4
    assert sensitivity_threshold(y, p1, 0.8) == pytest.approx(0.4)
    assert sensitivity_threshold(y, p1, 1.0) == pytest.approx(0.3)
    assert sensitivity_threshold(y, p1, 0.4) == pytest.approx(0.8)
    m = compute_metrics(y, np.stack([1 - p1, p1], axis=1), BINARY, threshold=0.4)
    assert m["sensitivity"] == pytest.approx(0.8) and m["specificity"] == pytest.approx(0.5)


def write_val_predictions(run_dir, y, p1):
    val_dir = run_dir / "eval_val"
    val_dir.mkdir(parents=True)
    pd.DataFrame({"image_id": [f"img{i}" for i in range(len(y))], "label": y,
                  "p_benign": 1 - p1, "p_melanoma": p1}).to_csv(val_dir / "predictions_val.csv", index=False)
    (val_dir / "metrics.json").write_text(json.dumps({"run": "toy", "checkpoint_epoch": 3}))


def run_script(*args):
    env = dict(os.environ, PYTHONPATH=str(PROJECT_DIR / "src"))
    return subprocess.run([sys.executable, *args], cwd=PROJECT_DIR, env=env, capture_output=True, text=True)


def test_select_threshold_writes_the_file_and_refuses_after_test(tmp_path):
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])
    p1 = np.array([0.1, 0.2, 0.5, 0.7, 0.3, 0.4, 0.6, 0.8, 0.9])
    write_val_predictions(tmp_path, y, p1)
    r = run_script("scripts/select_threshold.py", "--run", str(tmp_path))
    assert r.returncode == 0, r.stderr
    out = json.loads((tmp_path / "eval_val" / "threshold.json").read_text())
    assert out["checkpoint_epoch"] == 3
    assert out["thresholds"]["youden"]["threshold"] == pytest.approx(youden_threshold(y, p1))
    t = out["thresholds"]["sensitivity_0.80"]
    assert t["threshold"] == pytest.approx(0.4) and t["n_predicted_positive"] == 6
    assert t["val"]["sensitivity"] == pytest.approx(0.8)

    (tmp_path / "eval").mkdir()  # the test set has been evaluated
    r = run_script("scripts/select_threshold.py", "--run", str(tmp_path))
    assert r.returncode != 0 and "test set has been evaluated" in r.stderr


def test_binary_test_evaluation_needs_the_fixed_threshold(tmp_path):
    data = asdict(DataConfig.from_yaml(PROJECT_DIR / "configs" / "data" / "isic2020.yaml"))
    model = asdict(ModelConfig.from_yaml(PROJECT_DIR / "configs" / "model" / "convnext_t.yaml"))
    (tmp_path / "config.json").write_text(json.dumps({"data": data, "model": model}))
    r = run_script("scripts/evaluate.py", "--run", str(tmp_path))
    assert r.returncode != 0 and "threshold.json not found" in r.stderr
    assert not (tmp_path / "eval").exists()  # stopped before writing anything


def toy_binary(seed=0, n=400):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.2).astype(int)
    weak = np.clip(0.5 + 0.1 * (y - 0.5) + rng.normal(0, 0.2, n), 0, 1)
    strong = np.clip(0.5 + 0.8 * (y - 0.5) + rng.normal(0, 0.2, n), 0, 1)
    return y, np.stack([1 - weak, weak], 1), np.stack([1 - strong, strong], 1)


def test_paired_bootstrap_of_a_model_against_itself_is_zero():
    y, weak, _ = toy_binary()
    res = paired_bootstrap(y, weak, weak, {"argmax": None, "cut": 0.5}, {"argmax": None, "cut": 0.5}, n_boot=100)
    for r in res.values():
        assert r["diff"] == 0 and r["diff_ci"] == {"low": 0.0, "high": 0.0} and r["p_two_sided"] == 1.0


def test_paired_bootstrap_finds_a_clearly_better_model():
    y, weak, strong = toy_binary()
    res = paired_bootstrap(y, weak, strong, {"cut": 0.5}, {"cut": 0.5}, n_boot=200)
    assert res["auc"]["diff_ci"]["low"] > 0 and res["average_precision"]["diff_ci"]["low"] > 0
    assert res["auc"]["p_two_sided"] < 0.01
    assert res["auc"]["a"] == pytest.approx(compute_metrics(y, weak, BINARY)["auc"])
    assert res["cut_sensitivity"]["b"] == pytest.approx(compute_metrics(y, strong, BINARY, 0.5)["sensitivity"])


def write_test_run(run_dir, y, probs, cutoff):
    write_val_predictions(run_dir, y, probs[:, 1])
    thr = {"thresholds": {"sensitivity_0.80": {"threshold": cutoff}}}
    (run_dir / "eval_val" / "threshold.json").write_text(json.dumps(thr))
    sha = hashlib.sha256((run_dir / "eval_val" / "threshold.json").read_bytes()).hexdigest()
    (run_dir / "eval").mkdir()
    (run_dir / "eval" / "metrics.json").write_text(json.dumps({"run": "toy_a", "val_threshold_file": {"sha256": sha}}))
    pd.DataFrame({"image_id": [f"img{i}" for i in range(len(y))], "label": y, "p_benign": probs[:, 0],
                  "p_melanoma": probs[:, 1]}).to_csv(run_dir / "eval" / "predictions.csv", index=False)


def test_compare_paired_checks_image_order_and_threshold_files(tmp_path):
    y, weak, strong = toy_binary(n=60)
    a, b = tmp_path / "a", tmp_path / "b"
    write_test_run(a, y, weak, 0.5)
    write_test_run(b, y, strong, 0.5)
    pred = pd.read_csv(b / "eval" / "predictions.csv")
    pred.iloc[::-1].to_csv(b / "eval" / "predictions.csv", index=False)  # same images, other order
    r = run_script("scripts/compare_paired.py", "--runs", str(a), str(b), "--n-boot", "20")
    assert r.returncode != 0 and "same order" in r.stderr

    pred.to_csv(b / "eval" / "predictions.csv", index=False)
    (b / "eval_val" / "threshold.json").write_text(json.dumps({"thresholds": {"sensitivity_0.80": {"threshold": 0.4}}}))
    r = run_script("scripts/compare_paired.py", "--runs", str(a), str(b), "--n-boot", "20")
    assert r.returncode != 0 and "does not match" in r.stderr  # threshold changed after the test run


def toy_multiclass(seed=0, n=300, k=3):
    rng = np.random.default_rng(seed)
    y = np.arange(n) % k
    weak = rng.dirichlet(np.ones(k), n) + 0.3 * np.eye(k)[y]
    strong = rng.dirichlet(np.ones(k), n) + 2.0 * np.eye(k)[y]
    return y, weak / weak.sum(1, keepdims=True), strong / strong.sum(1, keepdims=True)


def test_paired_bootstrap_multiclass():
    classes = ["x", "y", "z"]
    y, weak, strong = toy_multiclass()
    same = paired_bootstrap_multiclass(y, weak, weak, classes, n_boot=50)
    assert all(r["diff"] == 0 and r["p_two_sided"] == 1.0 for r in same.values())
    res = paired_bootstrap_multiclass(y, weak, strong, classes, n_boot=200)
    assert res["macro_f1"]["a"] == pytest.approx(compute_metrics(y, weak, classes)["macro_f1"])
    assert res["macro_f1"]["diff_ci"]["low"] > 0 and res["macro_f1"]["p_two_sided"] < 0.01
    assert res["sensitivity_z"]["b"] == pytest.approx(compute_metrics(y, strong, classes)["per_class"]["z"]["sensitivity"])


def test_compare_paired_multiclass_needs_no_threshold_file(tmp_path):
    y, weak, strong = toy_multiclass(n=60)
    for name, probs in (("ham10000_a", weak), ("ham10000_b", strong)):
        (tmp_path / name / "eval").mkdir(parents=True)
        (tmp_path / name / "eval" / "metrics.json").write_text(json.dumps({"run": name}))
        frame = pd.DataFrame(probs, columns=["p_x", "p_y", "p_z"])
        frame.insert(0, "label", y)
        frame.insert(0, "image_id", [f"img{i}" for i in range(len(y))])
        frame.to_csv(tmp_path / name / "eval" / "predictions.csv", index=False)
    r = run_script("scripts/compare_paired.py", "--runs", str(tmp_path / "ham10000_a"), str(tmp_path / "ham10000_b"),
                   "--n-boot", "20")
    assert r.returncode == 0, r.stderr
    out_file = PROJECT_DIR / "outputs" / "comparisons" / "ham10000_test_ham10000_b_vs_ham10000_a.json"
    out = json.loads(out_file.read_text())
    out_file.unlink()
    out_file.with_suffix(".log").unlink()
    assert out["task"] == "multiclass" and out["classes"] == ["x", "y", "z"] and "macro_f1" in out["metrics"]


def test_multiclass_per_class_and_macro():
    classes = ["a", "b", "c"]
    y = np.array([0, 0, 1, 1, 2, 2])
    pred = np.array([0, 1, 1, 1, 2, 0])
    probs = np.eye(3)[pred] * 0.8 + 0.2 / 3
    m = compute_metrics(y, probs, classes)
    assert m["per_class"]["a"]["sensitivity"] == pytest.approx(0.5)
    assert m["per_class"]["b"]["sensitivity"] == pytest.approx(1.0)
    assert m["per_class"]["c"]["sensitivity"] == pytest.approx(0.5)
    assert m["macro_sensitivity"] == pytest.approx(2 / 3)
    assert m["balanced_accuracy"] == pytest.approx(2 / 3)
    # specificity of "a": negatives are the 4 non-a samples, one of them predicted a -> 3/4
    assert m["per_class"]["a"]["specificity"] == pytest.approx(3 / 4)


def test_auc_nan_when_class_missing():
    y = np.array([0, 0, 1, 1])
    probs = np.full((4, 3), 1 / 3)
    assert math.isnan(compute_metrics(y, probs, ["a", "b", "c"])["auc"])


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    p1 = np.clip(y * 0.3 + rng.random(300) * 0.7, 0, 1)
    probs = np.stack([1 - p1, p1], axis=1)
    point = compute_metrics(y, probs, BINARY)["auc"]
    ci = bootstrap_ci(y, probs, BINARY, n_boot=200)["auc"]
    assert ci["low"] <= point <= ci["high"] and ci["n_valid"] == 200


def test_warmup_cosine_schedule():
    assert warmup_cosine_factor(0, 10, 100, 0.01) == pytest.approx(0.1)
    assert warmup_cosine_factor(9, 10, 100, 0.01) == pytest.approx(1.0)
    assert warmup_cosine_factor(10, 10, 100, 0.01) == pytest.approx(1.0)
    assert warmup_cosine_factor(55, 10, 100, 0.01) == pytest.approx(0.505)  # half-way: (1 + 0.01) / 2
    assert warmup_cosine_factor(100, 10, 100, 0.01) == pytest.approx(0.01)
