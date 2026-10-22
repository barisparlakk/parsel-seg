"""Tiling, GSD resampling and scene-based train/val/test splitting."""

import random
from collections.abc import Iterator

import cv2
import numpy as np


def tile_origins(length: int, tile: int, stride: int) -> list[int]:
    """Start offsets covering [0, length). The last tile is shifted back to end at the edge."""
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] + tile < length:
        starts.append(length - tile)
    return starts


def pad_to(arr: np.ndarray, height: int, width: int, value: int) -> np.ndarray:
    """Pad bottom/right with a constant so arr is at least (height, width)."""
    pad_h, pad_w = max(0, height - arr.shape[0]), max(0, width - arr.shape[1])
    if not (pad_h or pad_w):
        return arr
    pad = [(0, pad_h), (0, pad_w)] + [(0, 0)] * (arr.ndim - 2)
    return np.pad(arr, pad, constant_values=value)


def resample(image: np.ndarray, mask: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """Rescale an image/mask pair by `scale` (area/linear for imagery, nearest for labels)."""
    if scale == 1.0:
        return image, mask
    h, w = mask.shape
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return (
        cv2.resize(image, size, interpolation=interp),
        cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST),
    )


def iter_tiles(
    image: np.ndarray, mask: np.ndarray, tile: int, overlap: int, ignore_index: int
) -> Iterator[tuple[int, int, np.ndarray, np.ndarray]]:
    """Yield (row, col, image_tile, mask_tile). Edge tiles smaller than `tile` are padded
    with black imagery and ignore_index labels."""
    if not 0 <= overlap < tile:
        raise ValueError("overlap must be in [0, tile)")
    stride = tile - overlap
    h, w = mask.shape
    for r in tile_origins(h, tile, stride):
        for c in tile_origins(w, tile, stride):
            img_t = pad_to(image[r : r + tile, c : c + tile], tile, tile, 0)
            msk_t = pad_to(mask[r : r + tile, c : c + tile], tile, tile, ignore_index)
            yield r, c, img_t, msk_t


def assign_splits(
    scenes: list[str], ratios: tuple[float, float, float] = (0.7, 0.15, 0.15), seed: int = 42
) -> dict[str, str]:
    """Assign whole scenes to train/val/test so no scene appears in two splits."""
    scenes = sorted(set(scenes))
    random.Random(seed).shuffle(scenes)
    n = len(scenes)
    n_val = max(1, round(n * ratios[1])) if n >= 3 else 0
    n_test = max(1, round(n * ratios[2])) if n >= 3 else 0
    out = {}
    for i, scene in enumerate(scenes):
        out[scene] = "val" if i < n_val else "test" if i < n_val + n_test else "train"
    return out
