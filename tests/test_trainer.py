"""End-to-end check of the training loop on a tiny synthetic task (CPU, a few seconds)."""
import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from skin_lesion_xai.config import TrainConfig
from skin_lesion_xai.train.trainer import Trainer, predict
from skin_lesion_xai.utils.checkpoint import load_checkpoint


class ToyDataset(Dataset):
    """Class 1 images are bright, class 0 images are dark: trivially learnable."""

    def __init__(self, n: int, seed: int):
        g = torch.Generator().manual_seed(seed)
        self.labels = torch.randint(0, 2, (n,), generator=g)
        self.images = torch.randn(n, 3, 16, 16, generator=g) * 0.3 + (self.labels.float() * 2 - 1)[:, None, None, None]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {"image": self.images[i], "label": self.labels[i], "image_id": f"img{i}"}


class ToyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Linear(8, 2)

    def head_parameters(self):
        return self.head.parameters()

    def set_grad_checkpointing(self, enable=True):
        pass

    def forward(self, x):
        return self.head(self.body(x))


def test_fit_learns_and_saves_loadable_checkpoint(tmp_path):
    torch.manual_seed(0)
    cfg = TrainConfig(epochs=4, batch_size=16, accum_steps=2, num_workers=0, warmup_epochs=1,
                      backbone_lr=1e-2, head_lr=1e-2, amp=False, ema_decay=0.9, early_stop_patience=10)
    model = ToyNet()
    trainer = Trainer(model, cfg, ["dark", "bright"], "binary", tmp_path / "run", torch.device("cpu"))
    train_loader = DataLoader(ToyDataset(128, 0), batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(ToyDataset(64, 1), batch_size=32)
    summary = trainer.fit(train_loader, val_loader)

    history = pd.read_csv(tmp_path / "run" / "history.csv")
    assert len(history) == 4 and summary["monitor"] == "auc"
    assert history["train_loss"].iloc[-1] < history["train_loss"].iloc[0]
    assert summary["best_val_score"] > 0.9  # the toy task is trivially separable

    reloaded = ToyNet()
    meta = load_checkpoint(tmp_path / "run" / "best.pt", reloaded)
    assert meta["epoch"] == summary["best_epoch"]
    out = predict(reloaded, val_loader, torch.device("cpu"), amp=False)
    assert out["probs"].shape == (64, 2) and len(out["image_ids"]) == 64


class Interrupted(Exception):
    pass


class InterruptingDataset(ToyDataset):
    """Raises after `stop_after` samples, like a run killed mid-epoch."""

    def __init__(self, n: int, seed: int, stop_after: int):
        super().__init__(n, seed)
        self.stop_after, self.calls = stop_after, 0

    def __getitem__(self, i):
        self.calls += 1
        if self.calls > self.stop_after:
            raise Interrupted
        return super().__getitem__(i)


def _resume_setup(tmp_path, name, train_ds):
    torch.manual_seed(0)
    cfg = TrainConfig(epochs=4, batch_size=16, accum_steps=2, num_workers=0, warmup_epochs=1,
                      backbone_lr=1e-2, head_lr=1e-2, amp=False, ema_decay=0.9, early_stop_patience=10)
    trainer = Trainer(ToyNet(), cfg, ["dark", "bright"], "binary", tmp_path / name, torch.device("cpu"))
    return trainer, DataLoader(train_ds, batch_size=16, shuffle=True), DataLoader(ToyDataset(64, 1), batch_size=32)


def test_resume_matches_uninterrupted_run(tmp_path):
    full, train_loader, val_loader = _resume_setup(tmp_path, "full", ToyDataset(128, 0))
    full_summary = full.fit(train_loader, val_loader)

    # killed 10 samples into epoch 3: last.pt holds the state after epoch 2
    killed, train_loader, val_loader = _resume_setup(tmp_path, "resumed", InterruptingDataset(128, 0, 2 * 128 + 10))
    with pytest.raises(Interrupted):
        killed.fit(train_loader, val_loader)
    assert len(pd.read_csv(tmp_path / "resumed" / "history.csv")) == 2

    torch.manual_seed(123)  # a different RNG state before resuming must not matter
    resumed, train_loader, val_loader = _resume_setup(tmp_path, "resumed", ToyDataset(128, 0))
    resumed_summary = resumed.fit(train_loader, val_loader, resume_from=tmp_path / "resumed" / "last.pt")

    assert resumed_summary == full_summary
    full_hist = pd.read_csv(tmp_path / "full" / "history.csv").drop(columns=["seconds"])
    resumed_hist = pd.read_csv(tmp_path / "resumed" / "history.csv").drop(columns=["seconds"])
    pd.testing.assert_frame_equal(full_hist, resumed_hist)
    for name in ("best.pt", "last.pt"):
        a = torch.load(tmp_path / "full" / name, weights_only=False)
        b = torch.load(tmp_path / "resumed" / name, weights_only=False)
        for key in a["model"]:
            torch.testing.assert_close(a["model"][key], b["model"][key])
    for key, value in full.model.state_dict().items():
        torch.testing.assert_close(value, resumed.model.state_dict()[key])


def test_resume_refuses_old_format_and_finished_runs(tmp_path):
    from skin_lesion_xai.utils.checkpoint import save_checkpoint
    trainer, train_loader, val_loader = _resume_setup(tmp_path, "run", ToyDataset(128, 0))
    save_checkpoint(tmp_path / "old.pt", trainer.eval_model, epoch=1)
    with pytest.raises(ValueError, match="old-format"):
        trainer.fit(train_loader, val_loader, resume_from=tmp_path / "old.pt")

    trainer, train_loader, val_loader = _resume_setup(tmp_path, "run", ToyDataset(128, 0))
    trainer.fit(train_loader, val_loader)
    trainer, train_loader, val_loader = _resume_setup(tmp_path, "run", ToyDataset(128, 0))
    with pytest.raises(ValueError, match="already finished"):
        trainer.fit(train_loader, val_loader, resume_from=tmp_path / "run" / "last.pt")


def test_strict_loading_rejects_wrong_architecture(tmp_path):
    from skin_lesion_xai.utils.checkpoint import save_checkpoint
    save_checkpoint(tmp_path / "m.pt", ToyNet())
    with pytest.raises(RuntimeError):
        load_checkpoint(tmp_path / "m.pt", nn.Linear(8, 2))
