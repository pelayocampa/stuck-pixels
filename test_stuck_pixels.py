import numpy as np

from stuck_pixels import find_outliers


def test_finds_hot_and_dead_pixels_but_not_edges():
    rng = np.random.default_rng(0)
    raw = rng.normal(2000, 20, (40, 60)).astype(np.uint16)
    raw[:, 30:] += 3000  # a real image edge must not be flagged
    raw[10, 11] = 16000  # hot
    raw[25, 40] = 0  # dead
    raw[0, 0] = 16000  # corner
    assert find_outliers(raw, black=512, white=16383, threshold=0.25) == {(10, 11), (25, 40), (0, 0)}
