import csv

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def tile_dataset(tmp_path):
    """A tiny processed dataset: 3 train tiles, 1 val tile, 64x64, class ids 0-8 plus 255."""
    rng = np.random.default_rng(0)
    rows = []
    for split, n in [("train", 3), ("val", 1)]:
        for i in range(n):
            stem = f"{split}_{i}"
            (tmp_path / split / "images").mkdir(parents=True, exist_ok=True)
            (tmp_path / split / "masks").mkdir(parents=True, exist_ok=True)
            img = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
            msk = rng.integers(0, 9, (64, 64), dtype=np.uint8)
            msk[:8] = 255
            Image.fromarray(img).save(tmp_path / split / "images" / f"{stem}.png")
            Image.fromarray(msk).save(tmp_path / split / "masks" / f"{stem}.png")
            rows.append(
                {"split": split, "image": f"{split}/images/{stem}.png",
                 "mask": f"{split}/masks/{stem}.png", "scene": f"s{i}", "source": stem,
                 "row": 0, "col": 0, "valid_frac": 0.9, "gsd_m": 0.3}
            )  # fmt: skip
    manifest = tmp_path / "manifest.csv"
    with open(manifest, "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return manifest
