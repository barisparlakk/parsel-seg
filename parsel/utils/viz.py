"""Mask colouring and image/mask overlays."""

import numpy as np

from parsel.data.dataset import IMAGENET_MEAN, IMAGENET_STD

IGNORE_COLOR = (255, 255, 255)


def colorize(
    mask: np.ndarray, palette: np.ndarray, ignore_index: int = 255, ignore_color=IGNORE_COLOR
) -> np.ndarray:
    """Turn an (H, W) class-id mask into an (H, W, 3) uint8 RGB image."""
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    valid = mask != ignore_index
    out[valid] = palette[mask[valid]]
    out[~valid] = ignore_color
    return out


def overlay(
    image: np.ndarray,
    mask: np.ndarray,
    palette: np.ndarray,
    alpha: float = 0.5,
    ignore_index: int = 255,
) -> np.ndarray:
    """Blend a coloured mask over an RGB uint8 image. Ignored pixels keep the raw image."""
    colour = colorize(mask, palette, ignore_index)
    blended = (1 - alpha) * image.astype(np.float32) + alpha * colour.astype(np.float32)
    out = image.copy()
    valid = mask != ignore_index
    out[valid] = blended[valid].astype(np.uint8)
    return out


def denormalize(
    image, mean: tuple[float, ...] = IMAGENET_MEAN, std: tuple[float, ...] = IMAGENET_STD
) -> np.ndarray:
    """Invert dataset normalisation: (3, H, W) tensor/array -> (H, W, 3) uint8."""
    arr = np.asarray(image, dtype=np.float32).transpose(1, 2, 0)
    arr = arr * np.array(std, dtype=np.float32) + np.array(mean, dtype=np.float32)
    return (np.clip(arr, 0, 1) * 255).round().astype(np.uint8)
