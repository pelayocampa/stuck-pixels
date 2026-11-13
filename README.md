# stuck-pixels

Scan a folder of Canon CR3 raw files and report sensor pixels that look stuck (always bright) or dead (always dark).

## How it works

1. For each file, the raw Bayer data is read with [rawpy](https://github.com/letmaik/rawpy) (LibRaw) and each pixel is compared to the median of its 8 same-colour neighbours.
2. Pixels that deviate by more than `--threshold` of the sensor range are outliers for that file.
3. A defect sits at the same coordinate in every shot, so only coordinates flagged in at least `--min-ratio` of the files are reported. This filters out stars, highlights and other real detail.

Best results: many shots, ideally including some dark frames (lens cap on, a few seconds of exposure).

## Usage

```sh
uv sync
uv run stuck_pixels.py /path/to/cr3/folder
uv run stuck_pixels.py /path/to/cr3/folder --threshold 0.15 --min-ratio 0.8 --recursive
```

Output is CSV (`row,col,files`) on stdout in raw sensor coordinates; progress goes to stderr, so `> stuck.csv` gives a clean file.

## Tests

```sh
uv run pytest
```
