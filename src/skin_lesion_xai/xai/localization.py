"""Localisation scores of a heatmap against a ground-truth lesion mask.

Every map is first upsampled to the image size (bilinear) and scaled to [0, 1] by its maximum.
  - pixel IoU and Dice of the map binarised at 15% of its maximum (threshold of Selvaraju et al. 2017,
    Grad-CAM §4.1; Dice as WA-NET Eq. 16);
  - IoU over a threshold sweep, and its mean ("iou_auc"), so the result does not hinge on one threshold;
  - box IoU: box of the largest connected region of the binarised map vs the box of the mask
    (Grad-CAM §4.1 localisation protocol; proposal slide 7);
  - pointing game: is the map's maximum inside the lesion (Grad-CAM §4.3);
  - energy in mask: share of the map's total value inside the lesion (threshold-free);
  - boundary energy: share of the map's total value on a band around the lesion border, the border taken
    from the mask with a Sobel filter as in WA-NET's edge supervision.
Reference maps (a centred Gaussian and a uniform map) are scored the same way, so a method only counts as
localising the lesion if it beats them; HAM10000 lesions are large and mostly centred.
An all-zero map (Grad-CAM whose weighted sum is negative everywhere) highlights nothing and scores 0 on every
metric, so averages over images always include it (and eval_xai.py reports the share of empty maps).
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F

METRICS = ("iou", "dice", "iou_auc", "bbox_iou", "pointing", "energy_in_mask", "boundary_energy")


def upsample(maps: np.ndarray, size: int) -> np.ndarray:
    """(N, h, w) raw maps -> (N, size, size) float32 in [0, 1] (bilinear, then divided by each map's max)."""
    t = torch.as_tensor(np.asarray(maps, dtype=np.float32))[:, None]
    up = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)[:, 0].clamp_min_(0)
    peak = up.flatten(1).amax(dim=1).clamp_min(1e-12)[:, None, None]
    return (up / peak).numpy()


def binarize(heatmap: np.ndarray, threshold: float) -> np.ndarray:
    """Pixels at or above threshold * max; empty for an all-zero map."""
    peak = heatmap.max()
    if peak <= 0:
        return np.zeros(heatmap.shape, dtype=bool)
    return heatmap >= threshold * peak


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else float("nan")


def dice(a: np.ndarray, b: np.ndarray) -> float:
    total = a.sum() + b.sum()
    return float(2 * np.logical_and(a, b).sum() / total) if total else float("nan")


def largest_component(binary: np.ndarray) -> np.ndarray:
    """Largest 8-connected region (empty input -> empty output)."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    if count <= 1:
        return np.zeros(binary.shape, dtype=bool)
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == biggest


def bbox(binary: np.ndarray) -> tuple[int, int, int, int] | None:
    """(y0, x0, y1, x1), inclusive; None if empty."""
    ys, xs = np.nonzero(binary)
    if len(ys) == 0:
        return None
    return int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())


def bbox_iou(a: tuple | None, b: tuple | None) -> float:
    if a is None or b is None:
        return 0.0 if (a is None) != (b is None) else float("nan")
    iy = max(0, min(a[2], b[2]) - max(a[0], b[0]) + 1)
    ix = max(0, min(a[3], b[3]) - max(a[1], b[1]) + 1)
    inter = iy * ix
    area = lambda r: (r[2] - r[0] + 1) * (r[3] - r[1] + 1)
    return float(inter / (area(a) + area(b) - inter))


def boundary_band(mask: np.ndarray, half_width: int) -> np.ndarray:
    """Pixels within half_width of the lesion border (Sobel edges of the mask, then dilated)."""
    m = mask.astype(np.float32)
    edges = np.hypot(cv2.Sobel(m, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(m, cv2.CV_32F, 0, 1, ksize=3)) > 0
    kernel = np.ones((2 * half_width + 1, 2 * half_width + 1), np.uint8)
    return cv2.dilate(edges.astype(np.uint8), kernel) > 0


def score_map(heatmap: np.ndarray, mask: np.ndarray, threshold: float, sweep: np.ndarray,
              band: np.ndarray, pointing: bool = True) -> dict:
    """All scores of one [0, 1] heatmap against one boolean mask (same H x W). NaN if the mask is empty.
    "iou_sweep" holds the IoU at each sweep threshold (not a per-image CSV column)."""
    mask = mask.astype(bool)
    if not mask.any():
        return ({k: float("nan") for k in METRICS} | {"map_empty": bool(heatmap.max() <= 0),
                                                       "iou_sweep": np.full(len(sweep), np.nan)})
    binary = binarize(heatmap, threshold)
    swept = heatmap[None] >= sweep[:, None, None] * max(heatmap.max(), 1e-12)
    inter = np.logical_and(swept, mask).sum(axis=(1, 2))
    union = np.logical_or(swept, mask).sum(axis=(1, 2))
    total = heatmap.sum()
    out = {
        "iou": iou(binary, mask),
        "dice": dice(binary, mask),
        "iou_sweep": inter / union,
        "iou_auc": float(np.mean(inter / union)),
        "bbox_iou": bbox_iou(bbox(largest_component(binary)), bbox(mask)),
        "pointing": float(mask.flat[int(np.argmax(heatmap))]) if pointing else float("nan"),
        "energy_in_mask": float((heatmap * mask).sum() / total) if total > 0 else float("nan"),
        "boundary_energy": float((heatmap * band).sum() / total) if total > 0 else float("nan"),
        "map_empty": bool(heatmap.max() <= 0),
    }
    if heatmap.max() <= 0:  # nothing highlighted: no overlap, no peak, no energy anywhere
        out.update(iou=0.0, dice=0.0, iou_auc=0.0, bbox_iou=0.0, pointing=0.0, energy_in_mask=0.0,
                   boundary_energy=0.0, iou_sweep=np.zeros(len(sweep)))
    return out


def centre_gaussian(size: int, sigma_frac: float) -> np.ndarray:
    """Isotropic Gaussian at the image centre, std = sigma_frac * size, max 1."""
    coords = np.arange(size, dtype=np.float32) - (size - 1) / 2
    g = np.exp(-(coords ** 2) / (2 * (sigma_frac * size) ** 2))
    return np.outer(g, g)


def reference_maps(size: int, sigma_frac: float) -> dict[str, np.ndarray]:
    """Image-independent maps every method is compared against."""
    return {"ref_centre": centre_gaussian(size, sigma_frac), "ref_uniform": np.ones((size, size), np.float32)}
