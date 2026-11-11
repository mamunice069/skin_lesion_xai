"""Trainer with the mask-guided losses of xai/mask_guidance.py (Part B, EXPLORATORY; scripts/finetune_mask.py).

Same loop, AMP, accumulation, clipping, EMA and model selection (val macro-F1) as Trainer; only the batch loss
changes, and the mean of each loss term per epoch is written to train.log and mask_terms.csv.
"""
from __future__ import annotations

from collections import defaultdict

import pandas as pd
import torch

from skin_lesion_xai.train.trainer import Trainer
from skin_lesion_xai.xai.mask_guidance import MaskGuidedLoss


class MaskGuidedTrainer(Trainer):
    def __init__(self, *args, lam: float, mu: float, noise_std: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_loss = MaskGuidedLoss(self.model, self.loss_fn, lam, mu, noise_std)
        self._terms: dict[str, list[float]] = defaultdict(list)
        self._epoch_rows: list[dict] = []

    def _batch_loss(self, batch: dict) -> torch.Tensor:
        x = batch["image"].to(self.device, non_blocking=True)
        y = batch["label"].to(self.device, non_blocking=True)
        masks = batch["mask"].to(self.device, non_blocking=True)
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
            loss, parts = self.mask_loss(x, y, masks)
        for k, v in parts.items():
            self._terms[k].append(v)
        return loss

    def _train_epoch(self, loader, scheduler) -> float:
        self._terms.clear()
        loss = super()._train_epoch(loader, scheduler)
        row = {"epoch": len(self._epoch_rows) + 1} | {k: float(sum(v) / len(v)) for k, v in self._terms.items()}
        self._epoch_rows.append(row)
        pd.DataFrame(self._epoch_rows).to_csv(self.out_dir / "mask_terms.csv", index=False)
        self.log.info("loss terms (epoch means): %s", "  ".join(f"{k} {v:.4f}" for k, v in row.items() if k != "epoch"))
        return loss
