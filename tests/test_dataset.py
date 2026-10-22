import numpy as np
import pytest
import torch

from parsel.data.dataset import TileDataset, build_transforms, class_pixel_counts


def test_item_shapes_and_dtypes(tile_dataset):
    item = TileDataset(tile_dataset, "train", build_transforms(True, tile_size=64))[0]
    assert item["image"].shape == (3, 64, 64)
    assert item["image"].dtype == torch.float32
    assert item["mask"].shape == (64, 64)
    assert item["mask"].dtype == torch.int64


def test_class_ids_in_range(tile_dataset):
    ds = TileDataset(tile_dataset, "train", build_transforms(True, tile_size=64))
    for i in range(len(ds)):
        ids = torch.unique(ds[i]["mask"])
        assert all((v < 9) or v == 255 for v in ids.tolist())


def test_split_selection(tile_dataset):
    assert len(TileDataset(tile_dataset, "train")) == 3
    assert len(TileDataset(tile_dataset, "val")) == 1


def test_unknown_split_raises(tile_dataset):
    with pytest.raises(ValueError):
        TileDataset(tile_dataset, "test")


def test_scale_augmentation_keeps_tile_size(tile_dataset):
    ds = TileDataset(tile_dataset, "train", build_transforms(True, tile_size=64))
    for _ in range(20):
        assert ds[0]["image"].shape == (3, 64, 64)
        assert ds[0]["mask"].shape == (64, 64)


def test_val_transform_is_deterministic(tile_dataset):
    ds = TileDataset(tile_dataset, "val")
    assert torch.equal(ds[0]["image"], ds[0]["image"])


def test_class_pixel_counts_exclude_ignore(tile_dataset):
    counts = class_pixel_counts(tile_dataset, "train", 9)
    assert counts.shape == (9,)
    assert counts.sum().item() == 3 * (64 * 64 - 8 * 64)
    assert np.all(counts.numpy() > 0)
