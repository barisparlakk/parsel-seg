"""Evaluate a trained run on a split: metrics, confusion matrix and worst predictions.

Writes runs/<run>/eval_<split>.json and, under docs/figures/, a normalised confusion matrix
and a gallery of the worst tiles (image | ground truth | prediction).

Usage:
    python scripts/evaluate.py --run runs/unet_r34 --split test
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from parsel.data.dataset import TileDataset, build_transforms
from parsel.models.factory import build_model, count_parameters
from parsel.train import resolve_device
from parsel.utils.classes import Taxonomy
from parsel.utils.metrics import BoundaryIoU, ConfusionMatrix
from parsel.utils.viz import colorize, denormalize

ROOT = Path(__file__).resolve().parents[1]


def load_model(run: Path, ckpt: str, num_classes: int, device: torch.device):
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    model_cfg = {**cfg["model"], "encoder_weights": None}  # weights come from the checkpoint
    model = build_model(model_cfg, num_classes)
    state = torch.load(run / ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    return cfg, model.to(device).eval(), state.get("epoch")


def tile_score(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> float:
    """Mean IoU over the classes present in this tile's ground truth (NaN if none)."""
    cm = ConfusionMatrix(num_classes)
    cm.update(pred, target)
    present = cm.present()
    return float(np.nanmean(cm.iou()[present])) if present.any() else float("nan")


def plot_confusion(mat: np.ndarray, names: list[str], path: Path, title: str) -> None:
    keep = np.flatnonzero((mat.sum(0) > 0) | (mat.sum(1) > 0))
    sub = mat[np.ix_(keep, keep)].astype(float)
    norm = sub / np.maximum(sub.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    labels = [names[i] for i in keep]
    ax.set_xticks(range(len(keep)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(keep)), labels)
    for i in range(len(keep)):
        for j in range(len(keep)):
            ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if norm[i, j] > 0.5 else "black")  # fmt: skip
    ax.set_xlabel("predicted")
    ax.set_ylabel("ground truth")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def plot_worst(dataset, model, device, worst, palette, path: Path, title: str) -> None:
    fig, axes = plt.subplots(len(worst), 3, figsize=(9, 3 * len(worst)))
    for row, (score, idx) in zip(axes, worst, strict=True):
        item = dataset[idx]
        with torch.no_grad():
            pred = model(item["image"][None].to(device)).argmax(1)[0].cpu().numpy()
        img = denormalize(item["image"].numpy())
        gt = item["mask"].numpy().astype(np.uint8)
        pred_rgb = colorize(pred.astype(np.uint8), palette)
        for a, im in zip(row, (img, colorize(gt, palette), pred_rgb), strict=True):
            a.imshow(im[::2, ::2])
            a.axis("off")
        row[0].set_title(
            f"{Path(item['name']).stem[:34]}  mIoU {score:.2f}", fontsize=7, loc="left"
        )
    axes[0, 1].set_title("ground truth", fontsize=8)
    axes[0, 2].set_title("prediction", fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(path, dpi=60)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--checkpoint", default="best.pt")
    p.add_argument("--worst", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--boundary-width", type=int, default=3, help="boundary IoU band in pixels")
    p.add_argument("--figures", type=Path, default=ROOT / "docs" / "figures")
    args = p.parse_args()

    tax = Taxonomy()
    names = [c.name for c in tax.classes]
    device = resolve_device("auto")
    cfg, model, epoch = load_model(args.run, args.checkpoint, len(names), device)
    ds = TileDataset(ROOT / cfg["data"]["manifest"], args.split, build_transforms(False))
    loader = DataLoader(ds, args.batch_size, shuffle=False, num_workers=4)

    cm = ConfusionMatrix(len(names), tax.ignore_index)
    bd = BoundaryIoU(len(names), args.boundary_width, tax.ignore_index)
    scores = []
    with torch.no_grad():
        for batch in loader:
            pred = model(batch["image"].to(device)).argmax(1).cpu()
            cm.update(pred, batch["mask"])
            bd.update(pred.to(device), batch["mask"].to(device))
            scores += [
                tile_score(pr, tg, len(names)) for pr, tg in zip(pred, batch["mask"], strict=True)
            ]
    s = cm.summary()

    tag = f"{args.run.name}_{args.split}"
    result = {
        "run": args.run.name, "checkpoint": args.checkpoint, "epoch": epoch, "split": args.split,
        "tiles": len(ds), "miou": s["miou"], "mf1": s["mf1"], "pixel_acc": s["pixel_acc"],
        "boundary_miou": bd.miou(), "boundary_width": args.boundary_width,
        "params_m": count_parameters(model) / 1e6,
        "iou": {n: None if np.isnan(v) else float(v) for n, v in zip(names, s["iou"], strict=True)},
        "f1": {n: None if np.isnan(v) else float(v) for n, v in zip(names, s["f1"], strict=True)},
        "boundary_iou": {n: None if np.isnan(v) else float(v)
                         for n, v in zip(names, bd.iou(), strict=True)},
        "gt_pixels": {n: int(v) for n, v in zip(names, cm.mat.sum(1), strict=True)},
        "confusion": cm.mat.tolist(),
    }  # fmt: skip
    (args.run / f"eval_{args.split}.json").write_text(json.dumps(result, indent=2))

    print(f"{tag}: {len(ds)} tiles, checkpoint epoch {epoch}")
    print(f"mIoU {s['miou']:.4f}  mF1 {s['mf1']:.4f}  pixel acc {s['pixel_acc']:.4f}  "
          f"boundary mIoU (+-{args.boundary_width}px) {bd.miou():.4f}")  # fmt: skip
    print(f"{'class':<14}{'IoU':>8}{'F1':>8}{'GT pixels':>14}")
    for i, n in enumerate(names):
        iou = "-" if np.isnan(s["iou"][i]) else f"{s['iou'][i]:.4f}"
        f1 = "-" if np.isnan(s["f1"][i]) else f"{s['f1'][i]:.4f}"
        print(f"{n:<14}{iou:>8}{f1:>8}{int(cm.mat.sum(1)[i]):>14}")

    args.figures.mkdir(parents=True, exist_ok=True)
    plot_confusion(cm.mat, names, args.figures / f"{tag}_confusion.png",
                   f"{args.run.name} on {args.split}: row-normalised confusion")  # fmt: skip
    order = [(sc, i) for i, sc in enumerate(scores) if not np.isnan(sc)]
    worst = sorted(order)[: args.worst]
    plot_worst(ds, model, device, worst, tax.palette(), args.figures / f"{tag}_worst.png",
               f"{args.run.name} on {args.split}: {len(worst)} worst tiles by mIoU")  # fmt: skip
    print(f"Wrote figures to {args.figures}")


if __name__ == "__main__":
    main()
