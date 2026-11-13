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


def reference_outliers(raw, black, white, threshold):
    """The original full-median implementation, kept as the ground truth for the fast path."""
    data = raw.astype(np.float32)
    hits = set()
    for dy in (0, 1):
        for dx in (0, 1):
            plane = data[dy::2, dx::2]
            padded = np.pad(plane, 1, mode="edge")
            h, w = plane.shape
            neighbours = np.stack([
                padded[1 + oy:1 + oy + h, 1 + ox:1 + ox + w]
                for oy in (-1, 0, 1) for ox in (-1, 0, 1) if (oy, ox) != (0, 0)
            ])
            diff = np.abs(plane - np.median(neighbours, axis=0))
            for r, c in zip(*np.nonzero(diff > threshold * (white - black))):
                hits.add((int(r) * 2 + dy, int(c) * 2 + dx))
    return hits


def test_fast_path_matches_full_median():
    rng = np.random.default_rng(1)
    for seed_threshold in (0.02, 0.1, 0.25):
        # Noisy frame with edges, clusters of defects, saturated and zero pixels: the hard cases.
        raw = rng.normal(3000, 400, (301, 403)).clip(0, 16383).astype(np.uint16)
        raw[:, 200:] = 15000
        idx = rng.integers(0, raw.size, 3000)
        raw.flat[idx] = rng.choice([0, 16383, 9000], idx.size)
        assert find_outliers(raw, 512, 16383, seed_threshold) == reference_outliers(raw, 512, 16383, seed_threshold)
