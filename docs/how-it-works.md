# How it works

## The sensor is a mosaic

Each photosite on a camera sensor measures only one colour, through a tiny red, green or blue filter. On most sensors the filters repeat in a 2x2 tile called a Bayer pattern:

```
R G R G R G
G B G B G B
R G R G R G
G B G B G B
```

Every tile holds one red, one blue and two greens (our eyes are most sensitive to green). The order differs between cameras (RGGB, BGGR, GRBG, GBRG), which does not matter here.

## Four planes, not three

The raw data is split by position in the 2x2 tile, `data[dy::2, dx::2]` for each `dy, dx` in `(0, 1)`. That gives four planes: red, blue, and two separate greens, usually called G1 (greens on the red rows) and G2 (greens on the blue rows).

The greens are kept apart for two reasons:

- Each plane is then a regular grid, so "the 8 nearest same-colour neighbours" is just a 3x3 window on the plane (a 5x5 area on the sensor).
- G1 and G2 often read slightly differently because of crosstalk from their red or blue neighbours ("green imbalance"). Raw converters correct this before demosaicing; uncorrected, it can show as a fine maze pattern in flat areas. Comparing a pixel only with neighbours of the same plane avoids treating that offset as a defect.

Sensors without a 2x2 pattern (Fujifilm X-Trans uses 6x6, Sigma Foveon stacks colours, linear DNGs are already demosaiced) are detected via `raw_pattern` / `raw_type` and skipped.

## Local median, per pixel

A pixel is an outlier in one file when it differs from the median of its 8 same-plane neighbours by more than `--threshold` of the sensor range (white level minus black level).

The median is local: one value per pixel, from its own neighbourhood. A global median per channel would flag every bright sky or white wall; a stuck pixel only stands out from its immediate neighbours. The median, rather than the mean, keeps one bright neighbour (another defect, a star) from hiding the pixel.

## Voting across files

A real defect sits at the same sensor coordinate in every shot; image detail does not. Coordinates are counted across all files and only those flagged in at least `--min-ratio` of them are reported. More files, and dark frames (lens cap on, a few seconds of exposure), make the result more reliable.

## Why it is fast: min/max prefilter

Computing a median for all ~24 million pixels of a frame is slow (about 3 s per 24 MP frame). The code avoids it with a guarantee: the median of the neighbours always lies between their minimum and maximum. So a pixel far from the median is necessarily far from the minimum (if brighter) or the maximum (if darker).

1. For every pixel, compute the min and max of its neighbours with `np.minimum` / `np.maximum` (cheap).
2. Keep only pixels where `pixel - min > limit` or `max - pixel > limit`. This can keep false candidates but never drops a real one.
3. Compute the exact median only for those candidates.

Results are identical to the full-median version, about 8x faster on a synthetic 24 MP frame. `test_fast_path_matches_full_median` keeps the original implementation as a reference and checks both agree on frames with edges, defect clusters, and clipped pixels.
