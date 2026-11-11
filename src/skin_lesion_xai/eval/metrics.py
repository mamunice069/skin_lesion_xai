"""Classification metrics.

Proposal slide 9: accuracy, sensitivity, F1 (HAM10000); AUC-ROC (primary, ISIC 2020), specificity;
precision-recall curves. Added: balanced accuracy, MCC and Cohen's kappa (reported by Mahesh et al. §4.3
and GC-ViT-E §3.5), average precision, and bootstrap 95% CIs (GC-ViT-E §3.5).

Predictions are argmax over class probabilities, except for binary tasks when a decision threshold on
P(class 1) is given (chosen on the validation split by scripts/select_threshold.py, never on test).
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    roc_auc_score,
    roc_curve,
)


def predict_labels(probs: np.ndarray, threshold: float | None = None) -> np.ndarray:
    if threshold is not None:
        if probs.shape[1] != 2:
            raise ValueError("a decision threshold only applies to binary tasks")
        return (probs[:, 1] >= threshold).astype(int)
    return probs.argmax(axis=1)


def _auc_and_ap(y_true: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    """Binary: on P(class 1). Multiclass: one-vs-rest, macro-averaged. NaN if a class is missing in y_true."""
    n_classes = probs.shape[1]
    if len(np.unique(y_true)) < n_classes:
        return float("nan"), float("nan")
    if n_classes == 2:
        return float(roc_auc_score(y_true, probs[:, 1])), float(average_precision_score(y_true, probs[:, 1]))
    onehot = np.eye(n_classes)[y_true]
    auc = roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
    ap = average_precision_score(onehot, probs, average="macro")
    return float(auc), float(ap)


def compute_metrics(y_true: np.ndarray, probs: np.ndarray, class_names: list[str],
                    threshold: float | None = None) -> dict:
    y_true = np.asarray(y_true)
    probs = np.asarray(probs, dtype=np.float64)
    n_classes = len(class_names)
    y_pred = predict_labels(probs, threshold)
    labels = list(range(n_classes))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tp = np.diag(cm).astype(float)
    fn = cm.sum(axis=1) - tp
    fp = cm.sum(axis=0) - tp
    tn = cm.sum() - tp - fn - fp
    with np.errstate(invalid="ignore", divide="ignore"):
        sensitivity = tp / (tp + fn)  # recall per class
        specificity = tn / (tn + fp)
    f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    auc, ap = _auc_and_ap(y_true, probs)

    out = {
        "n": int(len(y_true)),
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_sensitivity": float(np.nanmean(sensitivity)),
        "macro_specificity": float(np.nanmean(specificity)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "auc": auc,
        "average_precision": ap,
        "per_class": {
            name: {"sensitivity": float(sensitivity[i]), "specificity": float(specificity[i]),
                   "f1": float(f1[i]), "support": int(cm[i].sum())}
            for i, name in enumerate(class_names)
        },
        "confusion_matrix": cm.tolist(),
    }
    if n_classes == 2:  # positive class = 1 (melanoma): sensitivity = TPR, specificity = TNR
        out["sensitivity"] = float(sensitivity[1])
        out["specificity"] = float(specificity[1])
    return out


def youden_threshold(y_true: np.ndarray, p_positive: np.ndarray) -> float:
    """Threshold maximising sensitivity + specificity - 1 (Youden's J). Use on the validation split."""
    fpr, tpr, thresholds = roc_curve(y_true, p_positive)
    best = int(np.argmax(tpr - fpr))
    return float(min(thresholds[best], 1.0))


def sensitivity_threshold(y_true: np.ndarray, p_positive: np.ndarray, min_sensitivity: float) -> float:
    """Highest threshold whose sensitivity is >= min_sensitivity (the best specificity among those).
    Use on the validation split."""
    fpr, tpr, thresholds = roc_curve(y_true, p_positive, drop_intermediate=False)
    best = int(np.argmax(tpr >= min_sensitivity))  # thresholds fall and tpr rises along the curve
    return float(min(thresholds[best], 1.0))


BOOTSTRAP_METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "macro_sensitivity",
                     "macro_specificity", "mcc", "kappa", "auc", "average_precision")


def bootstrap_ci(y_true: np.ndarray, probs: np.ndarray, class_names: list[str], threshold: float | None = None,
                 n_boot: int = 1000, alpha: float = 0.05, seed: int = 0) -> dict:
    """Percentile bootstrap CIs over test samples. Resamples where a metric is undefined are skipped for it."""
    y_true, probs = np.asarray(y_true), np.asarray(probs)
    rng = np.random.default_rng(seed)
    values = {k: [] for k in BOOTSTRAP_METRICS}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        m = compute_metrics(y_true[idx], probs[idx], class_names, threshold)
        for k in BOOTSTRAP_METRICS:
            if not np.isnan(m[k]):
                values[k].append(m[k])
    return {
        k: {"low": float(np.quantile(v, alpha / 2)), "high": float(np.quantile(v, 1 - alpha / 2)), "n_valid": len(v)}
        if v else {"low": float("nan"), "high": float("nan"), "n_valid": 0}
        for k, v in values.items()
    }


def _binary_scores(y_true: np.ndarray, probs: np.ndarray, thresholds: dict) -> dict:
    """AUC, AP, and sensitivity / specificity at each named threshold (None = argmax)."""
    out = {"auc": float(roc_auc_score(y_true, probs[:, 1])),
           "average_precision": float(average_precision_score(y_true, probs[:, 1]))}
    pos = y_true == 1
    for name, thr in thresholds.items():
        pred = predict_labels(probs, thr) == 1
        out[f"{name}_sensitivity"] = float(pred[pos].mean())
        out[f"{name}_specificity"] = float((~pred[~pos]).mean())
    return out


def paired_bootstrap(y_true: np.ndarray, probs_a: np.ndarray, probs_b: np.ndarray, thresholds_a: dict,
                     thresholds_b: dict, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0) -> dict:
    """Paired percentile bootstrap of model B minus model A on the same binary test samples.

    Each resample draws one set of indices and scores both models on it, so the difference keeps the
    correlation between their errors. thresholds_a / thresholds_b map the same operating-point names to each
    model's own fixed cutoff (None = argmax). Per metric: point values of A, B and B - A, percentile CIs of
    all three, and a two-sided bootstrap p-value 2 * min(P(diff <= 0), P(diff >= 0)), capped at 1.
    Resamples without both classes are skipped (n_valid).
    """
    if thresholds_a.keys() != thresholds_b.keys():
        raise ValueError("both models need the same operating points")
    y_true, probs_a, probs_b = np.asarray(y_true), np.asarray(probs_a, float), np.asarray(probs_b, float)
    point_a, point_b = _binary_scores(y_true, probs_a, thresholds_a), _binary_scores(y_true, probs_b, thresholds_b)
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in point_a}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        a = _binary_scores(y_true[idx], probs_a[idx], thresholds_a)
        b = _binary_scores(y_true[idx], probs_b[idx], thresholds_b)
        for k in draws:
            draws[k].append((a[k], b[k]))
    return _summarise_pairs(point_a, point_b, draws, alpha)


PAIRED_MULTICLASS_METRICS = ("macro_f1", "balanced_accuracy", "accuracy", "mcc", "kappa", "auc", "average_precision")


def _multiclass_scores(y_true: np.ndarray, probs: np.ndarray, class_names: list[str]) -> dict:
    """Argmax metrics plus each class's sensitivity (sensitivity_<class>)."""
    m = compute_metrics(y_true, probs, class_names)
    return {k: m[k] for k in PAIRED_MULTICLASS_METRICS} | {
        f"sensitivity_{name}": m["per_class"][name]["sensitivity"] for name in class_names}


def paired_bootstrap_multiclass(y_true: np.ndarray, probs_a: np.ndarray, probs_b: np.ndarray, class_names: list[str],
                                n_boot: int = 1000, alpha: float = 0.05, seed: int = 0) -> dict:
    """paired_bootstrap for a multiclass task at argmax: the same resampled images for both models, B - A.
    Resamples missing a class are skipped (n_valid), so AUC and every per-class sensitivity are defined."""
    y_true, probs_a, probs_b = np.asarray(y_true), np.asarray(probs_a, float), np.asarray(probs_b, float)
    point_a, point_b = (_multiclass_scores(y_true, p, class_names) for p in (probs_a, probs_b))
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in point_a}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < len(class_names):
            continue
        a = _multiclass_scores(y_true[idx], probs_a[idx], class_names)
        b = _multiclass_scores(y_true[idx], probs_b[idx], class_names)
        for k in draws:
            draws[k].append((a[k], b[k]))
    return _summarise_pairs(point_a, point_b, draws, alpha)


def paired_bootstrap_per_image(values_a: np.ndarray, values_b: np.ndarray, n_boot: int = 1000, alpha: float = 0.05,
                               seed: int = 0, correct_a: np.ndarray | None = None,
                               correct_b: np.ndarray | None = None) -> dict:
    """Paired bootstrap of a per-image score, model B minus model A, on the same images (same index order).

    Statistic: the mean; with correct_a / correct_b given, the mean on correctly predicted images minus the
    mean on wrong ones, each model with its own correctness. Same output and p-value as paired_bootstrap;
    NaN values are skipped. A resample in which a model has no correct or no wrong image is skipped.
    """
    a, b = np.asarray(values_a, float), np.asarray(values_b, float)
    gap = correct_a is not None

    def stat(v, c, idx):
        v = v[idx]
        if not gap:
            return np.nanmean(v)
        c = np.asarray(c, bool)[idx]
        return np.nanmean(v[c]) - np.nanmean(v[~c]) if c.any() and (~c).any() else np.nan

    every = np.arange(len(a))
    point_a, point_b = {"value": stat(a, correct_a, every)}, {"value": stat(b, correct_b, every)}
    rng = np.random.default_rng(seed)
    draws = {"value": []}
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slices -> NaN, skipped below
        for _ in range(n_boot):
            idx = rng.integers(0, len(a), len(a))
            sa, sb = stat(a, correct_a, idx), stat(b, correct_b, idx)
            if not (np.isnan(sa) or np.isnan(sb)):
                draws["value"].append((sa, sb))
    return _summarise_pairs(point_a, point_b, draws, alpha)["value"]


def holm(p_values: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values (step-down, monotone, capped at 1)."""
    names = sorted(p_values, key=p_values.get)
    out, running = {}, 0.0
    for rank, name in enumerate(names):
        running = max(running, min(1.0, (len(names) - rank) * p_values[name]))
        out[name] = running
    return out


def _summarise_pairs(point_a: dict, point_b: dict, draws: dict, alpha: float) -> dict:
    def ci(v):
        return {"low": float(np.quantile(v, alpha / 2)), "high": float(np.quantile(v, 1 - alpha / 2))}

    out = {}
    nan_ci = {"low": float("nan"), "high": float("nan")}
    for k, pairs in draws.items():
        if not pairs:  # every resample was skipped (e.g. a model with no wrong prediction): no estimate
            out[k] = {"a": point_a[k], "b": point_b[k], "diff": point_b[k] - point_a[k], "a_ci": nan_ci,
                      "b_ci": nan_ci, "diff_ci": nan_ci, "p_two_sided": float("nan"), "n_valid": 0}
            continue
        ab = np.asarray(pairs)
        diff = ab[:, 1] - ab[:, 0]
        out[k] = {"a": point_a[k], "b": point_b[k], "diff": point_b[k] - point_a[k],
                  "a_ci": ci(ab[:, 0]), "b_ci": ci(ab[:, 1]), "diff_ci": ci(diff),
                  "p_two_sided": float(min(1.0, 2 * min((diff <= 0).mean(), (diff >= 0).mean()))),
                  "n_valid": len(diff)}
    return out
