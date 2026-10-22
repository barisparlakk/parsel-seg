"""Image and mask readers, plus pixel-size lookup for georeferenced rasters."""

from pathlib import Path

import numpy as np
import rasterio
from PIL import Image

TIFF_SUFFIXES = {".tif", ".tiff"}


def read_image(path: str | Path) -> np.ndarray:
    """Read an RGB uint8 image as (H, W, 3)."""
    path = Path(path)
    if path.suffix.lower() in TIFF_SUFFIXES:
        with rasterio.open(path) as src:
            arr = src.read([1, 2, 3])
        if arr.dtype != np.uint8:
            raise ValueError(f"{path}: expected uint8 imagery, got {arr.dtype}")
        return np.moveaxis(arr, 0, -1)
    return np.array(Image.open(path).convert("RGB"))


def read_mask(path: str | Path) -> np.ndarray:
    """Read a single-band class mask as (H, W) uint8."""
    path = Path(path)
    if path.suffix.lower() in TIFF_SUFFIXES:
        with rasterio.open(path) as src:
            return src.read(1)
    return np.array(Image.open(path))


def pixel_size_m(path: str | Path) -> float | None:
    """Ground sampling distance in metres, or None when the file is not georeferenced."""
    path = Path(path)
    if path.suffix.lower() not in TIFF_SUFFIXES:
        return None
    with rasterio.open(path) as src:
        if src.crs is None:
            return None
        gsd = abs(src.transform.a)
        if src.crs.is_geographic:
            lat = (src.bounds.top + src.bounds.bottom) / 2
            gsd = gsd * 111_320 * np.cos(np.radians(lat))
    return float(gsd)
