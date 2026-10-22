"""Class-balance statistics."""

import numpy as np


def median_frequency_weights(counts: np.ndarray) -> np.ndarray:
    """Median frequency balancing: w_c = median(freq of present classes) / freq_c.

    Classes with no pixels get weight 0.
    """
    freq = counts / counts.sum()
    present = freq > 0
    weights = np.zeros_like(freq, dtype=np.float64)
    weights[present] = np.median(freq[present]) / freq[present]
    return weights
