"""Evaluation figures: confusion matrix, ROC and precision-recall curves, training curves."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_curve


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(cm: list[list[int]], class_names: list[str], path: Path, title: str = "") -> None:
    cm = np.asarray(cm)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(1 + 0.8 * len(class_names), 0.8 + 0.8 * len(class_names)))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xticks(range(len(class_names)), class_names, rotation=45)
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title or "confusion matrix (count, row-normalised)", fontsize=9)
    _save(fig, path)


def _one_vs_rest(y_true, probs, class_names):
    if probs.shape[1] == 2:
        yield class_names[1], (y_true == 1).astype(int), probs[:, 1]
    else:
        for i, name in enumerate(class_names):
            yield name, (y_true == i).astype(int), probs[:, i]


def plot_roc_curves(y_true, probs, class_names, path: Path, title: str = "") -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    for name, y, p in _one_vs_rest(np.asarray(y_true), np.asarray(probs), class_names):
        if y.min() == y.max():
            continue
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, label=f"{name} (AUC {auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("false positive rate (1 - specificity)")
    ax.set_ylabel("true positive rate (sensitivity)")
    ax.set_title(title or "ROC (one-vs-rest)", fontsize=9)
    ax.legend(fontsize=7)
    _save(fig, path)


def plot_pr_curves(y_true, probs, class_names, path: Path, title: str = "") -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    for name, y, p in _one_vs_rest(np.asarray(y_true), np.asarray(probs), class_names):
        if y.max() == 0:
            continue
        precision, recall, _ = precision_recall_curve(y, p)
        ax.plot(recall, precision, label=f"{name} (prevalence {y.mean():.3f})")
    ax.set_xlabel("recall (sensitivity)")
    ax.set_ylabel("precision")
    ax.set_title(title or "precision-recall (one-vs-rest)", fontsize=9)
    ax.legend(fontsize=7)
    _save(fig, path)


def plot_history(history_csv: Path, path: Path, monitor: str) -> None:
    hist = pd.read_csv(history_csv)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(hist["epoch"], hist["train_loss"], label="train")
    ax1.plot(hist["epoch"], hist["val_loss"], label="val")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.legend()
    ax2.plot(hist["epoch"], hist[f"val_{monitor}"])
    ax2.set_xlabel("epoch")
    ax2.set_ylabel(f"val {monitor}")
    _save(fig, path)
