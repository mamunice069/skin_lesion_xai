"""Figure strips: image | image + lesion outline | one overlay per explanation map.

Each overlay shows the [0, 1] map (jet), the lesion outline (white) and the outline of the map binarised at
the localisation threshold (black), with that map's IoU and pointing-game hit in the title.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from skin_lesion_xai.data.transforms import IMAGENET_MEAN, IMAGENET_STD


def denormalize(image: torch.Tensor) -> np.ndarray:
    """(3, H, W) ImageNet-normalised tensor -> (H, W, 3) float RGB in [0, 1]."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (image.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def figure_indices(labels: np.ndarray, per_class: int, seed: int) -> list[int]:
    """Positions of the figure images: per_class per class, drawn with a fixed seed (the same in every run)."""
    rng = np.random.default_rng(seed)
    chosen = []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        chosen += sorted(rng.choice(idx, size=min(per_class, len(idx)), replace=False).tolist())
    return chosen


def _outline(ax, binary: np.ndarray, colour: str) -> None:
    if binary.any():
        ax.contour(binary.astype(float), levels=[0.5], colors=colour, linewidths=1.0)


def save_strip(path: Path, rgb: np.ndarray, mask: np.ndarray, maps: dict[str, np.ndarray],
               scores: dict[str, dict], threshold: float, title: str) -> None:
    """maps: {name: (H, W) in [0, 1]}; scores: {name: score_map(...) result}."""
    n = 2 + len(maps)
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 3.0))
    axes[0].imshow(rgb)
    axes[0].set_title("image", fontsize=8)
    axes[1].imshow(rgb)
    _outline(axes[1], mask, "white")
    axes[1].set_title(f"lesion mask ({mask.mean():.0%})", fontsize=8)
    for ax, (name, heatmap) in zip(axes[2:], maps.items()):
        ax.imshow(rgb)
        ax.imshow(heatmap, cmap="jet", alpha=0.45, vmin=0, vmax=1)
        _outline(ax, mask, "white")
        _outline(ax, heatmap >= threshold * max(heatmap.max(), 1e-12), "black")
        s = scores[name]
        hit = "" if np.isnan(s["pointing"]) else ("  hit" if s["pointing"] else "  miss")
        ax.set_title(f"{name}\nIoU {s['iou']:.2f}{hit}", fontsize=8)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
