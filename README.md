# stuck-pixels

Scan a folder of camera raw files (CR3, CR2, NEF, ARW, DNG, ORF, RW2, ... anything LibRaw reads) and report sensor pixels that look stuck (always bright) or dead (always dark).

## How it works

Files are recognised by content, not extension: every file in the folder is offered to LibRaw and non-raw files (JPEGs, sidecars) are ignored. Only 2x2 Bayer sensors are supported; Fujifilm X-Trans, Sigma Foveon and linear DNGs are skipped with a message.

1. For each file, the raw Bayer data is read with [rawpy](https://github.com/letmaik/rawpy) (LibRaw) and each pixel is compared to the median of its 8 same-colour neighbours.
2. Pixels that deviate by more than `--threshold` of the sensor range are outliers for that file.
3. A defect sits at the same coordinate in every shot, so only coordinates flagged in at least `--min-ratio` of the files are reported. This filters out stars, highlights and other real detail.

Details (Bayer planes, G1/G2, why the median is local, the speed trick): [docs/how-it-works.md](docs/how-it-works.md).

Best results: many shots (you get a warning below 10), ideally including some dark frames (lens cap on, a few seconds of exposure).

## Install

```sh
uv tool install git+https://github.com/pelayocampa/stuck-pixels
stuck-pixels /path/to/raws/folder
```

Upgrade with `uv tool upgrade stuck-pixels`. Works without uv too: `pipx install git+https://github.com/pelayocampa/stuck-pixels`.

## Usage (from a clone)

```sh
uv sync
uv run stuck_pixels.py /path/to/raws/folder
uv run stuck_pixels.py /path/to/raws/folder --threshold 0.15 --min-ratio 0.8 --recursive
uv run stuck_pixels.py /path/to/raws/folder --mark marks/ --mark-format jpg
```

Output is CSV on stdout: `row,col` in raw sensor coordinates, `files` (how many files flagged that pixel), `files_pct` and `file_names` (the files showing it, `;`-separated, relative to the folder). Progress, warnings and the summary go to stderr, so `> stuck.csv` gives a clean file.

With `--mark DIR`, every file that shows a stuck pixel gets its own folder `DIR/<subfolders>/<file name with extension>/`, holding:

- `<name>_marked.png`: the demosaiced image at full sensor size (same pixel grid as the CSV coordinates, no rotation) with each defect circled.
- `<name>_r<row>_c<col>.png`: a 33x33 pixel crop around each defect, enlarged 12x without smoothing, with the defect boxed.

PNG is lossless but large (about 50 MB per 24 MP image); `--mark-format jpg` is much smaller.

## Tests

```sh
uv run pytest
```
