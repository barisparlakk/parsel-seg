"""PyTorch dataset over the tiles written by scripts/make_tiles.py."""

import csv
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset

# Encoders are ImageNet-pretrained, so ImageNet statistics are the default. Dataset-specific
# channel statistics are written to configs/class_weights.yaml by scripts/class_stats.py.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IGNORE_INDEX = 255


def build_transforms(
    train: bool,
    tile_size: int = 512,
    mean: tuple[float, ...] = IMAGENET_MEAN,
    std: tuple[float, ...] = IMAGENET_STD,
) -> A.Compose:
    """Augmentation + normalisation pipeline. Validation/test only normalise."""
    normalise = [A.Normalize(mean=mean, std=std, max_pixel_value=255.0), ToTensorV2()]
    if not train:
        return A.Compose(normalise)
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomScale(scale_limit=(-0.25, 0.25), p=0.5),
            A.PadIfNeeded(
                tile_size,
                tile_size,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                fill_mask=IGNORE_INDEX,
            ),
            A.RandomCrop(tile_size, tile_size),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=15, val_shift_limit=0, p=0.3),
            *normalise,
        ]
    )


class TileDataset(Dataset):
    """Tiles of one split listed in a manifest.csv.

    Each item is a dict with `image` (3, H, W) float32, `mask` (H, W) int64 holding Parsel
    class ids (255 = ignore) and `name` (tile path relative to the manifest).
    """

    def __init__(self, manifest: str | Path, split: str, transform: A.Compose | None = None):
        self.root = Path(manifest).parent
        with open(manifest, newline="") as f:
            self.rows = [r for r in csv.DictReader(f) if r["split"] == split]
        if not self.rows:
            raise ValueError(f"No tiles for split {split!r} in {manifest}")
        self.transform = transform or build_transforms(train=(split == "train"))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        image = np.array(Image.open(self.root / row["image"]).convert("RGB"))
        mask = np.array(Image.open(self.root / row["mask"]))
        out = self.transform(image=image, mask=mask)
        return {"image": out["image"], "mask": out["mask"].long(), "name": row["image"]}


def class_pixel_counts(manifest: str | Path, split: str, num_classes: int) -> torch.Tensor:
    """Pixel count per class over a split (ignore_index excluded)."""
    counts = np.zeros(num_classes, dtype=np.int64)
    root = Path(manifest).parent
    with open(manifest, newline="") as f:
        for row in csv.DictReader(f):
            if row["split"] == split:
                m = np.array(Image.open(root / row["mask"])).ravel()
                counts += np.bincount(m[m != IGNORE_INDEX], minlength=num_classes)[:num_classes]
    return torch.from_numpy(counts)
