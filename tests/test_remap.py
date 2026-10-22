import numpy as np

from parsel.data.remap import load_lut, remap_mask


def test_openearthmap_mapping_targets():
    lut = load_lut("openearthmap")
    assert lut.shape == (256,) and lut.dtype == np.uint8
    raw = np.arange(9, dtype=np.uint8)
    # unlabeled, bareland, rangeland, developed, road, tree, water, agriculture, building
    assert remap_mask(raw, lut).tolist() == [255, 3, 7, 0, 5, 7, 6, 4, 1]


def test_unlisted_values_map_to_ignore():
    assert load_lut("openearthmap")[200] == 255


def test_loveda_mapping_targets():
    raw = np.arange(8, dtype=np.uint8)
    assert remap_mask(raw, load_lut("loveda")).tolist() == [255, 0, 1, 5, 6, 3, 7, 4]


def test_targets_are_valid_parsel_classes():
    from parsel.utils.classes import Taxonomy

    valid = {c.id for c in Taxonomy().classes} | {255}
    for name in ("openearthmap", "loveda"):
        assert set(np.unique(load_lut(name)).tolist()) <= valid
