"""Losses. Both tasks use num_classes logits (ISIC 2020 = 2 logits), so one loss path serves both."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from skin_lesion_xai.config import TrainConfig


class FocalLoss(nn.Module):
    """Multi-class focal loss (Lin et al., 2017): mean over the batch of w_y * (1 - p_y)^gamma * CE."""

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else None)
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, reduction="none", label_smoothing=self.label_smoothing)
        p_true = logits.softmax(dim=1).gather(1, target[:, None]).squeeze(1)
        loss = (1.0 - p_true).pow(self.gamma) * ce
        if self.weight is not None:
            loss = loss * self.weight[target]
        return loss.mean()


def build_loss(cfg: TrainConfig, class_weights: torch.Tensor | None = None) -> nn.Module:
    if cfg.loss == "ce":
        return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=cfg.label_smoothing)
    if cfg.loss == "focal":
        return FocalLoss(cfg.focal_gamma, class_weights, cfg.label_smoothing)
    raise ValueError(f"unknown loss {cfg.loss!r}")
