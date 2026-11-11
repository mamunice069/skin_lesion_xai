"""Training loop: AMP, gradient accumulation, gradient clipping, EMA, early stopping.

Recipe follows GC-ViT-E §3.4 (warmup + cosine, AMP, clipping, EMA, model selection on the EMA
weights, early stopping) and ViT App. B.1.1 (clip at global norm 1.0). Model selection uses the
validation split only; the test split is touched only by scripts/evaluate.py.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from timm.utils import ModelEmaV3
from torch import nn
from torch.utils.data import DataLoader

from skin_lesion_xai.config import TrainConfig
from skin_lesion_xai.eval.metrics import compute_metrics
from skin_lesion_xai.train.losses import build_loss
from skin_lesion_xai.train.optim import build_optimizer, build_scheduler
from skin_lesion_xai.utils.checkpoint import (load_training_state, save_checkpoint, save_training_state,
                                              set_rng_state)
from skin_lesion_xai.utils.device import gpu_memory_summary
from skin_lesion_xai.utils.logging import get_logger

HISTORY_METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "auc", "mcc")


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device, amp: bool = True,
            loss_fn: nn.Module | None = None) -> dict:
    """Softmax probabilities, labels and image ids for every sample (in loader order)."""
    model.eval()
    probs, labels, ids, loss_sum = [], [], [], 0.0
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            logits = model(x)
        logits = logits.float()
        if loss_fn is not None:
            loss_sum += loss_fn(logits, y).item() * len(y)
        probs.append(logits.softmax(dim=1).cpu().numpy())
        labels.append(y.cpu().numpy())
        ids += list(batch["image_id"])
    out = {"probs": np.concatenate(probs), "labels": np.concatenate(labels), "image_ids": ids}
    if loss_fn is not None:
        out["loss"] = loss_sum / len(out["labels"])
    return out


class Trainer:
    def __init__(self, model: nn.Module, cfg: TrainConfig, class_names: list[str], task: str,
                 out_dir: str | Path, device: torch.device, class_weights: torch.Tensor | None = None):
        self.cfg = cfg
        self.class_names = class_names
        self.device = device
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log = get_logger(f"train.{self.out_dir.name}", self.out_dir / "train.log")

        self.model = model.to(device)
        if cfg.grad_checkpointing:
            self.model.set_grad_checkpointing(True)
        self.loss_fn = build_loss(cfg, class_weights.to(device) if class_weights is not None else None).to(device)
        self.optimizer = build_optimizer(self.model, cfg)
        self.use_amp = cfg.amp and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        # EMA warmup ramps the decay up from 0, so early (and short smoke) runs are not dominated by init weights
        self.ema = ModelEmaV3(self.model, decay=cfg.ema_decay, use_warmup=True) if cfg.ema_decay > 0 else None
        self.monitor = cfg.monitor if cfg.monitor != "auto" else ("auc" if task == "binary" else "macro_f1")
        self.global_step = 0

    @property
    def eval_model(self) -> nn.Module:
        return self.ema.module if self.ema is not None else self.model

    def _batch_loss(self, batch: dict) -> torch.Tensor:
        """Training loss of one batch (subclasses add terms, e.g. train/mask_trainer.py)."""
        x = batch["image"].to(self.device, non_blocking=True)
        y = batch["label"].to(self.device, non_blocking=True)
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
            logits = self.model(x)
        return self.loss_fn(logits.float(), y)

    def _train_epoch(self, loader: DataLoader, scheduler) -> float:
        self.model.train()
        accum = self.cfg.accum_steps
        self.optimizer.zero_grad(set_to_none=True)
        loss_sum, n_seen = 0.0, 0
        for i, batch in enumerate(loader):
            y = batch["label"]
            loss = self._batch_loss(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss {loss.item()} at step {self.global_step}")
            self.scaler.scale(loss / accum).backward()
            loss_sum += loss.item() * len(y)
            n_seen += len(y)

            if (i + 1) % accum == 0 or (i + 1) == len(loader):
                if self.cfg.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                self.global_step += 1
                if self.ema is not None:
                    self.ema.update(self.model, step=self.global_step)
        return loss_sum / n_seen

    def _resume(self, path: str | Path, scheduler, steps_per_epoch: int) -> dict:
        """Restore everything saved in last.pt; returns the loop counters. Resumes at an epoch boundary."""
        ckpt = load_training_state(path)
        if ckpt.get("finished"):
            raise ValueError(f"{path}: run already finished at epoch {ckpt['epoch']}; nothing to resume")
        if ckpt["steps_per_epoch"] != steps_per_epoch:
            raise ValueError(f"{path}: saved with {ckpt['steps_per_epoch']} optimizer steps per epoch, now "
                             f"{steps_per_epoch}; the LR schedule would not line up")
        self.model.load_state_dict(ckpt["train_model"], strict=True)
        if self.ema is not None:
            self.ema.module.load_state_dict(ckpt["model"], strict=True)
        self.optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        self.scaler.load_state_dict(ckpt["scaler"])
        self.global_step = ckpt["global_step"]
        set_rng_state(ckpt["rng"])  # last, so the next epoch's shuffle matches an uninterrupted run
        return {k: ckpt[k] for k in ("epoch", "best_score", "best_epoch", "epochs_without_improvement")}

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, resume_from: str | Path | None = None) -> dict:
        cfg = self.cfg
        steps_per_epoch = math.ceil(len(train_loader) / cfg.accum_steps)
        scheduler = build_scheduler(self.optimizer, cfg, steps_per_epoch)
        best_path, last_path = self.out_dir / "best.pt", self.out_dir / "last.pt"
        best_score, best_epoch, epochs_without_improvement = -math.inf, 0, 0
        history, start_epoch = [], 1
        if resume_from is not None:
            state = self._resume(resume_from, scheduler, steps_per_epoch)
            best_score, best_epoch = state["best_score"], state["best_epoch"]
            epochs_without_improvement, start_epoch = state["epochs_without_improvement"], state["epoch"] + 1
            saved = pd.read_csv(self.out_dir / "history.csv")
            history = saved[saved["epoch"] <= state["epoch"]].to_dict("records")
            self.log.info("resumed from %s after epoch %d (best %s %.4f at epoch %d, global step %d)",
                          resume_from, state["epoch"], self.monitor, best_score, best_epoch, self.global_step)
        self.log.info("train %d / val %d samples, %d optimizer steps per epoch, effective batch %d, monitor %s",
                      len(train_loader.dataset), len(val_loader.dataset), steps_per_epoch,
                      cfg.effective_batch_size, self.monitor)

        for epoch in range(start_epoch, cfg.epochs + 1):
            start = time.time()
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
            train_loss = self._train_epoch(train_loader, scheduler)
            val = predict(self.eval_model, val_loader, self.device, self.use_amp, self.loss_fn)
            metrics = compute_metrics(val["labels"], val["probs"], self.class_names)
            score = metrics[self.monitor]

            row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val["loss"],
                   **{f"val_{k}": metrics[k] for k in HISTORY_METRICS},
                   "lr_backbone": self.optimizer.param_groups[0]["lr"],
                   "seconds": round(time.time() - start, 1), "gpu_peak": gpu_memory_summary(self.device)}
            history.append(row)
            pd.DataFrame(history).to_csv(self.out_dir / "history.csv", index=False)

            # NaN (e.g. a class missing from a tiny val set) never counts as an improvement,
            # but the first epoch is always saved so evaluate.py has a checkpoint to load.
            improved = (not math.isnan(score) and score > best_score) or not best_path.exists()
            if improved:
                best_score = score if not math.isnan(score) else best_score
                best_epoch, epochs_without_improvement = epoch, 0
                save_checkpoint(best_path, self.eval_model, epoch=epoch, monitor=self.monitor, score=score)
            else:
                epochs_without_improvement += 1
            stop_early = epochs_without_improvement >= cfg.early_stop_patience
            save_training_state(last_path, self.eval_model, train_model=self.model, optimizer=self.optimizer,
                                scheduler=scheduler, scaler=self.scaler, epoch=epoch, monitor=self.monitor,
                                score=score, global_step=self.global_step, best_score=best_score,
                                best_epoch=best_epoch, epochs_without_improvement=epochs_without_improvement,
                                steps_per_epoch=steps_per_epoch, finished=stop_early or epoch == cfg.epochs)

            self.log.info("epoch %d/%d  train_loss %.4f  val_loss %.4f  val_%s %.4f%s  (%.0fs, GPU peak %s)",
                          epoch, cfg.epochs, train_loss, val["loss"], self.monitor, score,
                          "  *best*" if improved else "", row["seconds"], row["gpu_peak"])
            if stop_early:
                self.log.info("early stopping: no improvement for %d epochs", epochs_without_improvement)
                break

        return {"best_epoch": best_epoch, "best_val_score": best_score, "monitor": self.monitor,
                "epochs_run": len(history)}
