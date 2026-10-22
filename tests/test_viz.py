import numpy as np

from parsel.utils.classes import Taxonomy
from parsel.utils.viz import colorize, denormalize, overlay


def test_colorize_uses_palette_and_ignore_color():
    pal = Taxonomy().palette()
    out = colorize(np.array([[1, 255]], dtype=np.uint8), pal)
    assert out.shape == (1, 2, 3) and out.dtype == np.uint8
    assert tuple(out[0, 0]) == tuple(pal[1]) and tuple(out[0, 1]) == (255, 255, 255)


def test_overlay_keeps_ignored_pixels():
    pal = Taxonomy().palette()
    img = np.full((1, 2, 3), 100, np.uint8)
    out = overlay(img, np.array([[1, 255]], dtype=np.uint8), pal, alpha=1.0)
    assert tuple(out[0, 0]) == tuple(pal[1]) and tuple(out[0, 1]) == (100, 100, 100)


def test_denormalize_roundtrip():
    from parsel.data.dataset import IMAGENET_MEAN, IMAGENET_STD

    img = np.random.default_rng(0).integers(0, 255, (8, 8, 3), dtype=np.uint8)
    norm = ((img / 255.0 - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)
    assert np.abs(denormalize(norm).astype(int) - img.astype(int)).max() <= 1
