"""Train a segmentation model from a YAML config.

Usage:
    python scripts/train.py --config configs/train/unet_r34.yaml
    python scripts/train.py --config configs/train/unet_r34.yaml --overfit 16 --epochs 40
    python scripts/train.py --config configs/train/unet_r34.yaml --resume runs/unet_r34/last.pt
"""

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from parsel.data.dataset import TileDataset, build_transforms
from parsel.losses import build_loss
from parsel.models.factory import build_model, count_parameters
from parsel.train import Trainer, resolve_device
from parsel.utils.classes import Taxonomy

ROOT = Path(__file__).resolve().parents[1]


def load_class_weights(cfg: dict, tax: Taxonomy) -> torch.Tensor | None:
    path = cfg.get("class_weights")
    if not path:
        return None
    raw = yaml.safe_load((ROOT / path).read_text())["weights"]
    w = np.array([raw[c.name] for c in tax.classes], dtype=np.float64)
    power = cfg.get("weight_power", 1.0)
    w = np.where(w > 0, w**power, 0.0)
    return torch.tensor(w, dtype=torch.float32)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--name", help="run name (default: config name)")
    p.add_argument("--epochs", type=int, help="override train.epochs")
    p.add_argument("--overfit", type=int, metavar="N", help="train and validate on N train tiles")
    p.add_argument("--max-batches", type=int, help="limit batches per epoch (smoke tests)")
    p.add_argument("--resume", type=Path)
    p.add_argument("--runs", type=Path, default=ROOT / "runs")
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    if args.max_batches:
        cfg["train"]["max_train_batches"] = args.max_batches
    if args.overfit:
        cfg["train"]["early_stop_patience"] = 0  # validation is on the training tiles
    name = args.name or (f"{cfg['name']}_overfit{args.overfit}" if args.overfit else cfg["name"])
    seed_everything(cfg.get("seed", 42))

    tax = Taxonomy()
    device = resolve_device(cfg["train"].get("device", "auto"))
    manifest = ROOT / cfg["data"]["manifest"]
    crop, bs, workers = (
        cfg["data"]["crop_size"],
        cfg["data"]["batch_size"],
        cfg["data"]["num_workers"],
    )

    if args.overfit:
        # Same tiles for train and validation, no augmentation: the model must memorise them.
        full = TileDataset(manifest, "train", build_transforms(train=False))
        idx = np.linspace(0, len(full) - 1, args.overfit).astype(int).tolist()
        train_ds = val_ds = Subset(full, idx)
        shuffle = True
    else:
        train_ds = TileDataset(manifest, "train", build_transforms(True, tile_size=crop))
        val_ds = TileDataset(manifest, "val", build_transforms(False))
        shuffle = True
    loader_kw = dict(num_workers=workers, persistent_workers=workers > 0)
    train_loader = DataLoader(train_ds, bs, shuffle=shuffle, drop_last=True, **loader_kw)
    val_loader = DataLoader(val_ds, 16, shuffle=False, **loader_kw)

    model = build_model(cfg["model"], len(tax.classes))
    loss_fn = build_loss(cfg["loss"], load_class_weights(cfg["loss"], tax), tax.ignore_index)
    out_dir = args.runs / name
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = copy.deepcopy(cfg)
    saved["run"] = {"overfit": args.overfit, "device": str(device)}
    (out_dir / "config.yaml").write_text(yaml.safe_dump(saved, sort_keys=False))

    print(f"run {name}: {len(train_ds)} train / {len(val_ds)} val tiles, device {device}, "
          f"{count_parameters(model) / 1e6:.1f}M params", flush=True)  # fmt: skip
    trainer = Trainer(
        model, loss_fn, train_loader, val_loader, [c.name for c in tax.classes],
        out_dir, cfg["train"], device, tax.ignore_index,
    )  # fmt: skip
    if args.resume:
        trainer.resume(args.resume)
    result = trainer.fit()
    print(json.dumps(result))


if __name__ == "__main__":
    main()
