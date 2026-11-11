"""Class-imbalance helpers: loss class weights and a class-balanced sampler."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


def class_counts(labels: np.ndarray, num_classes: int) -> np.ndarray:
    return np.bincount(np.asarray(labels), minlength=num_classes)


def class_weights(labels: np.ndarray, num_classes: int, power: float = 0.5) -> torch.Tensor:
    """(1 / frequency) ** power, scaled so the mean weight is 1. power=0.5 is sqrt inverse frequency."""
    counts = class_counts(labels, num_classes).astype(np.float64)
    if (counts == 0).any():
        raise ValueError(f"some classes have no samples: counts={counts.tolist()}")
    weights = (counts.sum() / counts) ** power
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def balanced_sampler(labels: np.ndarray, num_classes: int, power: float = 1.0, seed: int = 42) -> WeightedRandomSampler:
    """Draw samples with probability ~ (1 / class frequency) ** power, with replacement, one epoch = len(labels)."""
    labels = np.asarray(labels)
    per_class = class_weights(labels, num_classes, power).numpy()
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        torch.tensor(per_class[labels], dtype=torch.double), num_samples=len(labels), replacement=True, generator=generator
    )
