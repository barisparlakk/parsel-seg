"""Class remapping from a source dataset's raw mask values to Parsel class ids."""

from pathlib import Path

import numpy as np
import yaml

MAPPINGS_DIR = Path(__file__).resolve().parents[2] / "configs" / "mappings"


def load_lut(dataset: str | Path) -> np.ndarray:
    """Build a 256-entry uint8 lookup table from a mapping config.

    `dataset` is either a config name (e.g. "openearthmap") or a path to a YAML file.
    Raw values missing from the config map to ignore_index.
    """
    path = Path(dataset)
    if not path.suffix:
        path = MAPPINGS_DIR / f"{dataset}.yaml"
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    lut = np.full(256, cfg["ignore_index"], dtype=np.uint8)
    for raw, entry in cfg["mapping"].items():
        lut[int(raw)] = entry["target"]
    return lut


def remap_mask(mask: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Apply a lookup table to a uint8 mask."""
    return lut[mask]
