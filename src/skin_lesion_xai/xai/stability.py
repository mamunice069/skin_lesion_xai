"""Small input perturbations for the stability check: the explanation of an unchanged decision should not
change much. Images are normalised tensors (ImageNet mean / std, data/transforms.py); noise and brightness
act on the 0-1 pixel scale and are clipped there. A flipped image's map is flipped back before comparing.
"""
from __future__ import annotations

import numpy as np
import torch

from skin_lesion_xai.data.transforms import IMAGENET_MEAN, IMAGENET_STD


def perturbation_names(noise_std: float, brightness: list[float]) -> list[str]:
    return ["hflip", f"noise_{noise_std:g}"] + [f"brightness_{f:g}" for f in brightness]


def _pixels(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return x * std + mean, mean, std


def perturb(x: torch.Tensor, name: str, seeds: list[int] | None = None) -> torch.Tensor:
    """name from perturbation_names; seeds (one per image) make the noise reproducible."""
    if name == "hflip":
        return x.flip(-1)
    pix, mean, std = _pixels(x)
    if name.startswith("noise_"):
        sigma = float(name.split("_", 1)[1])
        noise = torch.stack([torch.randn(x.shape[1:], generator=torch.Generator().manual_seed(s)) for s in seeds])
        pix = pix + sigma * noise.to(x.device)
    elif name.startswith("brightness_"):
        pix = pix * float(name.split("_", 1)[1])
    else:
        raise ValueError(name)
    return (pix.clamp(0, 1) - mean) / std


def restore_map(maps: np.ndarray, name: str) -> np.ndarray:
    """Bring (B, h, w) maps of a perturbed image back to the original image's geometry."""
    return maps[..., ::-1].copy() if name == "hflip" else maps
