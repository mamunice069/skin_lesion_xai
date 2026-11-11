"""Checkpoints hold the model state_dict plus metadata. Loading is strict: a mismatch is an error,
never a warning (the Mahesh et al. repo silently explained an untrained model this way).

last.pt additionally holds the full training state (raw model, optimizer, scheduler, AMP scaler,
RNG states) so an interrupted run can resume; its "model" entry is the same as in best.pt."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

TRAINING_STATE_KEYS = ("train_model", "optimizer", "scheduler", "scaler", "rng")


def save_checkpoint(path: str | Path, model: nn.Module, **meta) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), **meta}, tmp)
    tmp.replace(path)  # atomic: an interrupted save never leaves a half-written checkpoint


def rng_state() -> dict:
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_training_state(path: str | Path, model: nn.Module, *, train_model: nn.Module,
                        optimizer: torch.optim.Optimizer, scheduler, scaler, **meta) -> None:
    """model = the weights used for evaluation (EMA); train_model = the weights being optimised."""
    save_checkpoint(path, model, train_model=train_model.state_dict(), optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(), scaler=scaler.state_dict(), rng=rng_state(), **meta)


def load_training_state(path: str | Path) -> dict:
    """Raw checkpoint dict for resuming; refuses checkpoints that only hold model weights."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    missing = [k for k in TRAINING_STATE_KEYS if k not in ckpt]
    if missing:
        raise ValueError(f"{path} is an old-format checkpoint without {missing}; cannot resume from it")
    return ckpt


def load_checkpoint(path: str | Path, model: nn.Module, map_location="cpu") -> dict:
    """Load weights into model (strict) and return the stored metadata."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt.pop("model"), strict=True)
    return ckpt
