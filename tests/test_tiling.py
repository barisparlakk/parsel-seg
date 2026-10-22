import numpy as np
import pytest

from parsel.data.tiling import assign_splits, iter_tiles, pad_to, resample, tile_origins


def test_tile_origins_cover_whole_length():
    for length, tile, stride in [(1024, 512, 512), (1000, 512, 448), (900, 512, 256)]:
        starts = tile_origins(length, tile, stride)
        assert starts[0] == 0 and starts[-1] + tile == length
        assert all(b - a <= stride for a, b in zip(starts, starts[1:], strict=False))


def test_tile_origins_smaller_than_tile():
    assert tile_origins(438, 512, 512) == [0]


def test_iter_tiles_count_and_shape():
    img, msk = np.zeros((1024, 1024, 3), np.uint8), np.zeros((1024, 1024), np.uint8)
    tiles = list(iter_tiles(img, msk, 512, 0, 255))
    assert len(tiles) == 4
    assert all(t[2].shape == (512, 512, 3) and t[3].shape == (512, 512) for t in tiles)


def test_small_image_is_padded_with_ignore():
    img, msk = np.ones((400, 300, 3), np.uint8), np.zeros((400, 300), np.uint8)
    ((_, _, img_t, msk_t),) = iter_tiles(img, msk, 512, 0, 255)
    assert img_t.shape == (512, 512, 3) and msk_t.shape == (512, 512)
    assert msk_t[0, 0] == 0 and msk_t[-1, -1] == 255 and img_t[-1, -1].sum() == 0


def test_overlap_must_be_smaller_than_tile():
    with pytest.raises(ValueError):
        list(iter_tiles(np.zeros((8, 8, 3), np.uint8), np.zeros((8, 8), np.uint8), 4, 4, 255))


def test_pad_to_noop_when_large_enough():
    a = np.zeros((10, 10), np.uint8)
    assert pad_to(a, 8, 8, 255) is a


def test_resample_keeps_label_values():
    msk = np.random.default_rng(0).integers(0, 9, (100, 100), dtype=np.uint8)
    img = np.zeros((100, 100, 3), np.uint8)
    img2, msk2 = resample(img, msk, 0.5)
    assert img2.shape == (50, 50, 3) and msk2.shape == (50, 50)
    assert set(np.unique(msk2)) <= set(np.unique(msk))


def test_assign_splits_is_scene_disjoint_and_deterministic():
    scenes = [f"city{i}" for i in range(40)] * 3  # repeated: many tiles per scene
    a, b = assign_splits(scenes, seed=1), assign_splits(scenes, seed=1)
    assert a == b
    assert len(a) == 40 and set(a.values()) == {"train", "val", "test"}
    assert sum(v == "val" for v in a.values()) == 6
