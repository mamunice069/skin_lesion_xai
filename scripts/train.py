"""Train one model on one dataset.

Usage:
  python scripts/train.py --data configs/data/ham10000.yaml --model configs/model/resnet50.yaml \
      --profile configs/train/local_4gb.yaml [--run-name NAME] [--epochs N] [--max-train N --max-val N]

Writes outputs/runs/<run-name>/: config.json, train.log, history.csv, best.pt, last.pt.
--max-train / --max-val take stratified subsets (smoke tests only; results are not meaningful).
--resume continues outputs/runs/<run-name>/ from last.pt (same arguments as the original run; any
config difference is refused).
"""
import argparse
import dataclasses
import json
from datetime import datetime

from torch.utils.data import DataLoader

from skin_lesion_xai.config import PROJECT_DIR, DataConfig, ModelConfig, TrainConfig
from skin_lesion_xai.data.datasets import build_dataset
from skin_lesion_xai.data.samplers import balanced_sampler, class_counts, class_weights
from skin_lesion_xai.data.transforms import build_transform
from skin_lesion_xai.models.factory import build_model
from skin_lesion_xai.train.trainer import Trainer
from skin_lesion_xai.utils.device import get_device
from skin_lesion_xai.utils.seed import seed_everything


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--train", default="configs/train/default.yaml", help="base recipe")
    p.add_argument("--profile", default="configs/train/local_4gb.yaml", help="hardware profile, overrides --train")
    p.add_argument("--run-name", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--backbone-lr", type=float, default=None, help="overrides the recipe's backbone_lr")
    p.add_argument("--max-train", type=int, default=None)
    p.add_argument("--max-val", type=int, default=None)
    p.add_argument("--hair-removed", action="store_true", help="train on the DullRazor cache (ablation)")
    p.add_argument("--resume", action="store_true", help="continue the existing --run-name from its last.pt")
    args = p.parse_args()
    if args.resume and args.run_name is None:
        p.error("--resume needs --run-name")
    return args


def config_differences(saved: dict, current: dict, prefix: str = "") -> list[str]:
    """Dotted paths of every setting that differs between two config snapshots."""
    diffs = []
    for key in sorted(set(saved) | set(current)):
        a, b = saved.get(key), current.get(key)
        if isinstance(a, dict) and isinstance(b, dict):
            diffs += config_differences(a, b, f"{prefix}{key}.")
        elif a != b:
            diffs.append(f"{prefix}{key}: saved {a!r}, now {b!r}")
    return diffs


def main() -> None:
    args = parse_args()
    data_cfg = DataConfig.from_yaml(args.data)
    if args.hair_removed:
        data_cfg.use_hair_removal = True
    model_cfg = ModelConfig.from_yaml(args.model)
    train_cfg = TrainConfig.from_yamls(args.train, args.profile, epochs=args.epochs, backbone_lr=args.backbone_lr)
    seed_everything(train_cfg.seed)

    smoke = args.max_train is not None or args.max_val is not None
    run_name = args.run_name or "_".join(filter(None, [
        data_cfg.name, model_cfg.name, "hair" if data_cfg.use_hair_removal else None,
        "smoke" if smoke else None, datetime.now().strftime("%Y%m%d-%H%M%S")]))
    out_dir = PROJECT_DIR / "outputs" / "runs" / run_name
    if args.resume:
        if not (out_dir / "last.pt").exists():
            raise SystemExit(f"--resume: {out_dir / 'last.pt'} not found")
        if (out_dir / "train_summary.json").exists():
            raise SystemExit(f"--resume: {out_dir} already finished (train_summary.json exists)")
    elif out_dir.exists():
        raise SystemExit(f"{out_dir} already exists; choose another --run-name (or pass --resume)")

    train_ds = build_dataset(data_cfg, "train", build_transform("train", data_cfg.image_size),
                             max_samples=args.max_train, seed=train_cfg.seed)
    val_ds = build_dataset(data_cfg, "val", build_transform("val", data_cfg.image_size),
                           max_samples=args.max_val, seed=train_cfg.seed)

    loss_weights = (class_weights(train_ds.labels, data_cfg.num_classes, train_cfg.class_weight_power)
                    if train_cfg.class_weight_power > 0 else None)
    sampler = (balanced_sampler(train_ds.labels, data_cfg.num_classes, seed=train_cfg.seed)
               if train_cfg.sampler == "balanced" else None)
    train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size, sampler=sampler, shuffle=sampler is None,
                              drop_last=True, num_workers=train_cfg.num_workers, pin_memory=True,
                              persistent_workers=train_cfg.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=train_cfg.batch_size, shuffle=False,
                            num_workers=train_cfg.val_num_workers, pin_memory=True,
                            persistent_workers=train_cfg.val_num_workers > 0)

    model = build_model(model_cfg, data_cfg.num_classes)
    trainer = Trainer(model, train_cfg, data_cfg.classes, data_cfg.task, out_dir, get_device(), loss_weights)

    config_snapshot = {
        "run_name": run_name, "smoke": smoke, "data_config": args.data, "model_config": args.model,
        "data": dataclasses.asdict(data_cfg), "model": dataclasses.asdict(model_cfg),
        "train": dataclasses.asdict(train_cfg),
        "train_class_counts": class_counts(train_ds.labels, data_cfg.num_classes).tolist(),
        "loss_class_weights": loss_weights.tolist() if loss_weights is not None else None,
        "max_train": args.max_train, "max_val": args.max_val,
    }
    config_path = out_dir / "config.json"
    if args.resume:
        saved = json.loads(config_path.read_text())
        resumed_at = saved.pop("resumed_at", [])
        # round-trip through JSON so tuples and lists compare equal
        diffs = config_differences(saved, json.loads(json.dumps(config_snapshot)))
        if diffs:
            raise SystemExit("--resume: config differs from the saved run:\n  " + "\n  ".join(diffs))
        config_snapshot["resumed_at"] = resumed_at + [datetime.now().isoformat(timespec="seconds")]
    config_path.write_text(json.dumps(config_snapshot, indent=2))

    summary = trainer.fit(train_loader, val_loader, resume_from=out_dir / "last.pt" if args.resume else None)
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    trainer.log.info("done: %s", summary)


if __name__ == "__main__":
    main()
