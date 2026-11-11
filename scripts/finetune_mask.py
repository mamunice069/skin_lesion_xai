"""Mask-guided fine-tuning of the cross-attention hybrid (Part B, EXPLORATORY): CE + lam * outside-mask share of
the fused CAM + mu * its noise consistency (xai/mask_guidance.py, train/mask_trainer.py).

Usage: python scripts/finetune_mask.py --lam 0.3 [--mu 0] [--source outputs/runs/ham10000_hybrid] [--run-name NAME]

Starts from <source>/best.pt (loaded strictly) with the source run's data and model config; recipe =
configs/train/default.yaml + local_4gb.yaml + mask_finetune.yaml. Masks are read for the train split only
(train transforms move image and mask together); val is loaded without masks, so model selection (best.pt =
best val macro-F1 epoch) sees no masks. Writes outputs/runs/<run-name>/ like train.py, plus mask_terms.csv;
config.json records the source checkpoint (epoch, SHA-256) and lam / mu / noise.
"""
import argparse
import dataclasses
import hashlib
import json

from torch.utils.data import DataLoader

from skin_lesion_xai.config import PROJECT_DIR, DataConfig, ModelConfig, TrainConfig
from skin_lesion_xai.data.datasets import build_dataset
from skin_lesion_xai.data.samplers import class_counts, class_weights
from skin_lesion_xai.data.transforms import build_transform
from skin_lesion_xai.models.factory import build_model
from skin_lesion_xai.train.mask_trainer import MaskGuidedTrainer
from skin_lesion_xai.utils.checkpoint import load_checkpoint
from skin_lesion_xai.utils.device import get_device
from skin_lesion_xai.utils.seed import seed_everything


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="outputs/runs/ham10000_hybrid")
    p.add_argument("--lam", type=float, required=True, help="weight of the outside-mask share")
    p.add_argument("--mu", type=float, default=0.0, help="weight of the noise-consistency term (0 = off)")
    p.add_argument("--noise-std", type=float, default=0.03, help="consistency noise, std on the 0-1 pixel scale")
    p.add_argument("--profile", default="configs/train/local_4gb.yaml")
    p.add_argument("--recipe", default="configs/train/mask_finetune.yaml")
    p.add_argument("--run-name", default=None)
    p.add_argument("--max-train", type=int, default=None, help="smoke test only")
    p.add_argument("--max-val", type=int, default=None, help="smoke test only")
    p.add_argument("--epochs", type=int, default=None, help="smoke test only")
    args = p.parse_args()

    source = PROJECT_DIR / args.source
    snap = json.loads((source / "config.json").read_text())
    data_cfg, model_cfg = DataConfig(**snap["data"]), ModelConfig(**snap["model"])
    train_cfg = TrainConfig.from_yamls("configs/train/default.yaml", args.profile, args.recipe, epochs=args.epochs)
    seed_everything(train_cfg.seed)
    smoke = args.max_train is not None or args.max_val is not None
    run_name = args.run_name or (f"{source.name}_mask_l{args.lam:g}_m{args.mu:g}" + ("_smoke" if smoke else ""))
    out_dir = PROJECT_DIR / "outputs" / "runs" / run_name
    if out_dir.exists():
        raise SystemExit(f"{out_dir} already exists; choose another --run-name")

    train_ds = build_dataset(data_cfg, "train", build_transform("train", data_cfg.image_size), return_mask=True,
                             max_samples=args.max_train, seed=train_cfg.seed)
    val_ds = build_dataset(data_cfg, "val", build_transform("val", data_cfg.image_size),
                           max_samples=args.max_val, seed=train_cfg.seed)
    loss_weights = (class_weights(train_ds.labels, data_cfg.num_classes, train_cfg.class_weight_power)
                    if train_cfg.class_weight_power > 0 else None)
    train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=True, drop_last=True,
                              num_workers=train_cfg.num_workers, pin_memory=True,
                              persistent_workers=train_cfg.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=train_cfg.batch_size, shuffle=False,
                            num_workers=train_cfg.val_num_workers, pin_memory=True,
                            persistent_workers=train_cfg.val_num_workers > 0)

    model = build_model(model_cfg, data_cfg.num_classes, pretrained=False)
    source_epoch = load_checkpoint(source / "best.pt", model)["epoch"]
    trainer = MaskGuidedTrainer(model, train_cfg, data_cfg.classes, data_cfg.task, out_dir, get_device(),
                                loss_weights, lam=args.lam, mu=args.mu, noise_std=args.noise_std)
    snapshot = {
        "run_name": run_name, "smoke": smoke, "data": dataclasses.asdict(data_cfg),
        "model": dataclasses.asdict(model_cfg), "train": dataclasses.asdict(train_cfg),
        "train_class_counts": class_counts(train_ds.labels, data_cfg.num_classes).tolist(),
        "loss_class_weights": loss_weights.tolist() if loss_weights is not None else None,
        "max_train": args.max_train, "max_val": args.max_val,
        "finetuned_from": {"run": source.name, "epoch": source_epoch,
                           "sha256": hashlib.sha256((source / "best.pt").read_bytes()).hexdigest()},
        "mask_guidance": {"lam": args.lam, "mu": args.mu, "noise_std": args.noise_std, "map": "fused CAM",
                          "masks": "train split only"},
    }
    (out_dir / "config.json").write_text(json.dumps(snapshot, indent=2))
    trainer.log.info("fine-tuning %s (epoch %d) with lam %g, mu %g", source.name, source_epoch, args.lam, args.mu)
    summary = trainer.fit(train_loader, val_loader)
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    trainer.log.info("done: %s", summary)


if __name__ == "__main__":
    main()
