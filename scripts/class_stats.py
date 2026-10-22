"""Class frequencies, median-frequency weights and channel statistics for a split.

Writes configs/class_weights.yaml. Weights follow median frequency balancing:
w_c = median(freq of present classes) / freq_c. Classes with no pixels get weight 0.

Usage:
    python scripts/class_stats.py --dataset openearthmap
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from parsel.data.dataset import class_pixel_counts
from parsel.utils.classes import Taxonomy
from parsel.utils.stats import median_frequency_weights

ROOT = Path(__file__).resolve().parents[1]


def channel_stats(manifest: Path, split: str, every: int = 5) -> tuple[list[float], list[float]]:
    """Per-channel mean/std in [0, 1] over every n-th tile of a split."""
    total = np.zeros(3)
    total_sq = np.zeros(3)
    n = 0
    with open(manifest, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == split][::every]
    for row in rows:
        img = np.array(Image.open(manifest.parent / row["image"]).convert("RGB")) / 255.0
        total += img.sum(axis=(0, 1))
        total_sq += (img**2).sum(axis=(0, 1))
        n += img.shape[0] * img.shape[1]
    mean = total / n
    std = np.sqrt(total_sq / n - mean**2)
    return mean.round(4).tolist(), std.round(4).tolist()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--processed", type=Path, default=ROOT / "data" / "processed")
    p.add_argument("--out", type=Path, default=ROOT / "configs" / "class_weights.yaml")
    args = p.parse_args()

    tax = Taxonomy()
    manifest = args.processed / args.dataset / "manifest.csv"
    counts = class_pixel_counts(manifest, args.split, len(tax.classes)).numpy()
    weights = median_frequency_weights(counts.astype(np.float64))
    mean, std = channel_stats(manifest, args.split)

    print(f"{'class':<14}{'pixels':>14}{'freq %':>9}{'weight':>9}")
    for c in tax.classes:
        print(
            f"{c.name:<14}{counts[c.id]:>14}{100 * counts[c.id] / counts.sum():>9.2f}"
            f"{weights[c.id]:>9.3f}"
        )
    out = {
        "dataset": args.dataset,
        "split": args.split,
        "note": "median frequency balancing; classes absent from the data have weight 0",
        "pixels": {c.name: int(counts[c.id]) for c in tax.classes},
        "weights": {c.name: round(float(weights[c.id]), 4) for c in tax.classes},
        "channel_mean": mean,
        "channel_std": std,
    }
    args.out.write_text(yaml.safe_dump(out, sort_keys=False), encoding="utf-8")
    print(f"channel mean {mean} std {std}\nWrote {args.out}")


if __name__ == "__main__":
    main()
