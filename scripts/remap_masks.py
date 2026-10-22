"""Convert raw source masks to Parsel class ids.

Writes full-size uint8 PNG masks to data/interim/<dataset>/<scene>/<name>.png using the
mapping in configs/mappings/<dataset>.yaml. Run before scripts/make_tiles.py.

Usage:
    python scripts/remap_masks.py --dataset openearthmap
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from parsel.data.io import read_mask
from parsel.data.remap import load_lut, remap_mask
from parsel.data.sources import DEFAULT_RAW, list_samples

ROOT = Path(__file__).resolve().parents[1]


def _work(args: tuple[Path, Path, np.ndarray]) -> tuple[str, np.ndarray]:
    src, dst, lut = args
    mask = read_mask(src)
    out = remap_mask(mask, lut)
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out).save(dst)
    return dst.name, np.bincount(mask.ravel(), minlength=256)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--dataset", required=True, choices=["openearthmap", "loveda"])
    p.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    p.add_argument("--out", type=Path, default=ROOT / "data" / "interim")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = p.parse_args()

    lut = load_lut(args.dataset)
    samples = list_samples(args.dataset, args.raw)
    if not samples:
        raise SystemExit(f"No image/mask pairs found for {args.dataset} under {args.raw}")
    jobs = [(s.mask, args.out / args.dataset / s.scene / f"{s.name}.png", lut) for s in samples]
    raw_counts = np.zeros(256, dtype=np.int64)
    with ProcessPoolExecutor(args.workers) as pool:
        for i, (_, counts) in enumerate(pool.map(_work, jobs, chunksize=16), 1):
            raw_counts += counts
            if i % 200 == 0 or i == len(jobs):
                print(f"\r{i}/{len(jobs)} masks", end="", flush=True)
    print()
    unmapped = [v for v in np.flatnonzero(raw_counts) if lut[v] == 255 and v not in (0,)]
    print(f"Wrote {len(jobs)} masks to {args.out / args.dataset}")
    print(f"Raw values present: {np.flatnonzero(raw_counts).tolist()}")
    if unmapped:
        print(f"WARNING: raw values mapped to ignore_index beyond value 0: {unmapped}")


if __name__ == "__main__":
    main()
