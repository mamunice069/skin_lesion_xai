"""AdamW with separate backbone / head learning rates, and per-step linear warmup + cosine decay."""
from __future__ import annotations

import math

import torch
from torch import nn

from skin_lesion_xai.config import TrainConfig
from skin_lesion_xai.models.factory import head_parameters


def build_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    """Four groups: {backbone, head} x {decay, no decay}. Biases and norm weights (1-D) get no weight decay."""
    head_ids = {id(p) for p in head_parameters(model)}
    groups = {(h, d): [] for h in (False, True) for d in (False, True)}
    for p in model.parameters():
        if p.requires_grad:
            groups[(id(p) in head_ids, p.ndim > 1)].append(p)
    param_groups = [
        {"params": params, "lr": cfg.head_lr if is_head else cfg.backbone_lr,
         "weight_decay": cfg.weight_decay if decay else 0.0}
        for (is_head, decay), params in groups.items() if params
    ]
    return torch.optim.AdamW(param_groups)


def warmup_cosine_factor(step: int, warmup_steps: int, total_steps: int, min_ratio: float) -> float:
    """LR multiplier: linear 0->1 over warmup_steps, then cosine 1->min_ratio until total_steps."""
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: TrainConfig, steps_per_epoch: int):
    """Stepped once per optimizer step (not per micro-batch)."""
    total = cfg.epochs * steps_per_epoch
    warmup = round(cfg.warmup_epochs * steps_per_epoch)
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: warmup_cosine_factor(step, warmup, total, cfg.min_lr_ratio)
    )
