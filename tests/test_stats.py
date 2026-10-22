import numpy as np

from parsel.utils.stats import median_frequency_weights


def test_rarer_classes_get_larger_weights():
    w = median_frequency_weights(np.array([700, 200, 100]))
    assert w[0] < w[1] < w[2]


def test_median_class_has_weight_one():
    assert median_frequency_weights(np.array([700, 200, 100]))[1] == 1.0


def test_absent_class_has_zero_weight():
    w = median_frequency_weights(np.array([500, 0, 500]))
    assert w[1] == 0 and w[0] == w[2] == 1.0
