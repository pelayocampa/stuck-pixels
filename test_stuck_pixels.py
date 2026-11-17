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


def test_cli_votes_across_files_and_keeps_stdout_pure_csv(tmp_path, monkeypatch, capsys):
    import sys
    import stuck_pixels

    for i in range(4):
        (tmp_path / f"img{i}.raw").write_bytes(b"x")
    # (5, 7) is flagged in all 4 files, (1, 1) in only one: a chance repeat that the vote must drop.
    hits = iter([{(5, 7), (1, 1)}, {(5, 7)}, {(5, 7)}, {(5, 7)}])
    monkeypatch.setattr(stuck_pixels, "scan_file", lambda path, threshold: (next(hits), (10, 10)))
    monkeypatch.setattr(sys, "argv", ["stuck-pixels", str(tmp_path)])

    assert stuck_pixels.main() == 0
    out, err = capsys.readouterr()
    assert out == "row,col,files,files_pct,file_names\n5,7,4,100.0,img0.raw;img1.raw;img2.raw;img3.raw\n"
    assert "only 4 raw files" in err


def test_mark_writes_full_size_image_and_zoom_crops(tmp_path):
    from PIL import Image
    from stuck_pixels import ZOOM_RADIUS, ZOOM_SCALE, save_marks

    rgb = np.zeros((100, 150, 3), np.uint8)
    rgb[50, 60] = 255
    save_marks(rgb, {(50, 60), (2, 3)}, tmp_path / "img", "png")

    assert Image.open(tmp_path / "img_marked.png").size == (150, 100)
    side = (2 * ZOOM_RADIUS + 1) * ZOOM_SCALE
    zoom = np.asarray(Image.open(tmp_path / "img_r50_c60.png"))
    assert zoom.shape[:2] == (side, side)
    centre = ZOOM_RADIUS * ZOOM_SCALE + ZOOM_SCALE // 2
    assert (zoom[centre, centre] == 255).all()  # the defect sits in the middle of the crop
    # A defect near the corner gets a clipped crop instead of an error.
    assert Image.open(tmp_path / "img_r2_c3.png").size == ((3 + ZOOM_RADIUS + 1) * ZOOM_SCALE, (2 + ZOOM_RADIUS + 1) * ZOOM_SCALE)
