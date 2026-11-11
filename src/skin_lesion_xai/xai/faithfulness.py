"""Deletion / insertion curves: does removing (adding) the pixels a map ranks highest change the prediction?

Petsiuk et al. 2018 (RISE), deletion and insertion metrics; external, not in Related_papers. Pixels are
ranked by the map (ties by pixel index), and in `steps` steps the top 1/steps, 2/steps, ... of them are
changed. Deletion fills them with 0 in normalised space (the dataset-mean colour); insertion starts from a
Gaussian-blurred copy and restores them. The score is the softmax probability of a fixed class (the one
predicted on the original image); the AUC over the fraction changed is the metric. Lower deletion AUC and
higher insertion AUC mean the map points at pixels the model relies on.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torchvision.transforms.functional import gaussian_blur


def pixel_order(maps: np.ndarray) -> torch.Tensor:
    """(B, H, W) maps -> (B, H*W) position of each pixel in the descending ranking (0 = highest)."""
    flat = torch.as_tensor(np.ascontiguousarray(maps), dtype=torch.float32).flatten(1)
    order = torch.argsort(-flat, dim=1, stable=True)
    rank = torch.empty_like(order)
    rank.scatter_(1, order, torch.arange(flat.shape[1]).expand_as(order).contiguous())
    return rank


@torch.no_grad()
def perturbation_curves(model: nn.Module, x: torch.Tensor, maps: np.ndarray, classes: torch.Tensor, steps: int,
                        mode: str, blur_kernel: int = 11, blur_sigma: float = 5.0, forward_batch: int = 64,
                        amp: bool = True) -> np.ndarray:
    """x (B, 3, H, W) normalised images; maps (B, H, W); classes (B,). Returns (B, steps + 1) probabilities,
    column k after changing round(k / steps * H * W) pixels."""
    if mode not in ("deletion", "insertion"):
        raise ValueError(mode)
    b, _, h, w = x.shape
    device = x.device
    rank = pixel_order(maps).to(device)
    counts = torch.round(torch.arange(steps + 1, device=device) / steps * h * w).long()
    start = gaussian_blur(x, [blur_kernel, blur_kernel], [blur_sigma, blur_sigma]) if mode == "insertion" else None
    img_idx = torch.arange(b, device=device).repeat_interleave(steps + 1)
    step_idx = torch.arange(steps + 1, device=device).repeat(b)
    scores = torch.empty(b * (steps + 1), device=device)
    for s in range(0, len(img_idx), forward_batch):
        bi, ki = img_idx[s:s + forward_batch], step_idx[s:s + forward_batch]
        changed = (rank[bi] < counts[ki][:, None]).view(-1, 1, h, w)
        if mode == "deletion":
            xi = torch.where(changed, torch.zeros((), device=device), x[bi])
        else:
            xi = torch.where(changed, x[bi], start[bi])
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            logits = model(xi)
        scores[s:s + forward_batch] = logits.float().softmax(dim=1).gather(1, classes[bi][:, None])[:, 0]
    return scores.view(b, steps + 1).cpu().numpy()


def curve_auc(curves: np.ndarray) -> np.ndarray:
    """(B, steps + 1) -> (B,) trapezoid area over the fraction of pixels changed (0..1)."""
    return np.trapezoid(curves, dx=1.0 / (curves.shape[1] - 1), axis=1)
