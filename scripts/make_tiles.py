"""Cut remapped scenes into fixed-size tiles and split them by scene.

Reads images from data/raw and remapped masks from data/interim (scripts/remap_masks.py),
writes data/processed/<dataset>/{train,val,test}/{images,masks}/*.png and manifest.csv.
Scenes (OpenEarthMap regions) are never split across train/val/test. Datasets with an
official split (LoveDA) keep it.

Usage:
    python scripts/make_tiles.py --dataset openearthmap --tile-size 512 --overlap 64
    python scripts/make_tiles.py --dataset openearthmap --target-gsd 0.5
"""

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from parsel.data.io import pixel_size_m, read_image, read_mask
from parsel.data.sources import DEFAULT_GSD, DEFAULT_RAW, Sample, list_samples
from parsel.data.tiling import assign_splits, iter_tiles, resample

ROOT = Path(__file__).resolve().parents[1]
IGNORE = 255
FIELDS = ["split", "image", "mask", "scene", "source", "row", "col", "valid_frac", "gsd_m"]


def _work(job: tuple) -> list[dict]:
    sample, mask_path, out_dir, split, cfg = job
    image, mask = read_image(sample.image), read_mask(mask_path)
    gsd = pixel_size_m(sample.image) or DEFAULT_GSD[cfg["dataset"]]
    if cfg["target_gsd"] and gsd:
        image, mask = resample(image, mask, gsd / cfg["target_gsd"])
        gsd = cfg["target_gsd"]
    rows = []
    for r, c, img_t, msk_t in iter_tiles(image, mask, cfg["tile"], cfg["overlap"], IGNORE):
        valid = float(np.mean(msk_t != IGNORE))
        if valid < cfg["min_valid"]:
            continue
        stem = f"{sample.scene}__{sample.name}__r{r}_c{c}"
        img_rel, msk_rel = f"{split}/images/{stem}.png", f"{split}/masks/{stem}.png"
        for rel in (img_rel, msk_rel):
            (out_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(img_t).save(out_dir / img_rel, compress_level=1)
        Image.fromarray(msk_t).save(out_dir / msk_rel)
        rows.append(
            dict(
                split=split, image=img_rel, mask=msk_rel, scene=sample.scene,
                source=sample.name, row=r, col=c, valid_frac=round(valid, 4),
                gsd_m=round(gsd, 4) if gsd else "",
            )
        )  # fmt: skip
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--dataset", required=True, choices=["openearthmap", "loveda"])
    p.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    p.add_argument("--interim", type=Path, default=ROOT / "data" / "interim")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "processed")
    p.add_argument("--tile-size", type=int, default=512)
    p.add_argument("--overlap", type=int, default=0, help="pixels shared between neighbours")
    p.add_argument("--target-gsd", type=float, default=None, help="resample to this m/pixel")
    p.add_argument(
        "--min-valid", type=float, default=0.5, help="drop tiles with less labelled area"
    )
    p.add_argument("--ratios", type=float, nargs=3, default=(0.7, 0.15, 0.15))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = p.parse_args()

    samples: list[Sample] = list_samples(args.dataset, args.raw)
    if not samples:
        raise SystemExit(f"No image/mask pairs found for {args.dataset} under {args.raw}")
    splits = assign_splits([s.scene for s in samples], tuple(args.ratios), args.seed)
    out_dir = args.out / args.dataset
    cfg = dict(
        dataset=args.dataset, tile=args.tile_size, overlap=args.overlap,
        target_gsd=args.target_gsd, min_valid=args.min_valid,
    )  # fmt: skip
    jobs = []
    for s in samples:
        mask_path = args.interim / args.dataset / s.scene / f"{s.name}.png"
        if not mask_path.exists():
            raise SystemExit(f"Missing {mask_path}; run scripts/remap_masks.py first")
        jobs.append((s, mask_path, out_dir, s.split_hint or splits[s.scene], cfg))

    rows = []
    with ProcessPoolExecutor(args.workers) as pool:
        for i, r in enumerate(pool.map(_work, jobs, chunksize=4), 1):
            rows.extend(r)
            if i % 100 == 0 or i == len(jobs):
                print(f"\r{i}/{len(jobs)} images, {len(rows)} tiles", end="", flush=True)
    print()
    with open(out_dir / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        w.writerows(rows)
    for split in ("train", "val", "test"):
        sel = [r for r in rows if r["split"] == split]
        scenes = {r["scene"] for r in sel}
        print(f"{split}: {len(sel)} tiles from {len(scenes)} scenes")


if __name__ == "__main__":
    main()
