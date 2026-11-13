"""Find stuck/hot/dead pixels across a folder of camera raw files.

A defective pixel sits at the same sensor coordinate in every shot, so we flag
outliers per file on the raw Bayer data and keep only coordinates that repeat
across many files. Real image detail (stars, specular highlights) rarely repeats.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import rawpy
from tqdm import tqdm


def find_outliers(raw: np.ndarray, black: float, white: float, threshold: float) -> set[tuple[int, int]]:
    """Return (row, col) of pixels that differ from their same-colour neighbours by > threshold of the range."""
    data = raw.astype(np.float32)
    limit = threshold * (white - black)
    hits: set[tuple[int, int]] = set()
    # Each 2x2 Bayer offset is a single-colour plane; compare pixels only to their own colour.
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
            for r, c in zip(*np.nonzero(diff > limit)):
                hits.add((int(r) * 2 + dy, int(c) * 2 + dx))
    return hits


def scan_file(path: Path, threshold: float) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    with rawpy.imread(str(path)) as raw:
        # The neighbour comparison assumes a 2x2 Bayer mosaic (rules out X-Trans, Foveon, linear DNG).
        if raw.raw_type != rawpy.RawType.Flat or raw.raw_pattern is None or raw.raw_pattern.shape != (2, 2):
            raise ValueError("not a 2x2 Bayer sensor, unsupported")
        data = raw.raw_image_visible
        black = float(np.mean(raw.black_level_per_channel))
        return find_outliers(data, black, float(raw.white_level), threshold), data.shape


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("folder", type=Path)
    ap.add_argument("--threshold", type=float, default=0.25,
                    help="min deviation from neighbours as fraction of sensor range (default 0.25)")
    ap.add_argument("--min-ratio", type=float, default=0.5,
                    help="fraction of files a pixel must be flagged in (default 0.5)")
    ap.add_argument("--recursive", action="store_true", help="also scan subfolders")
    args = ap.parse_args()

    if not args.folder.is_dir():
        ap.error(f"not a folder: {args.folder}")
    pattern = "**/*" if args.recursive else "*"
    # No extension list: LibRaw recognises raw formats by content, so every file is offered to it.
    files = sorted(p for p in args.folder.glob(pattern) if p.is_file() and not p.name.startswith("."))

    counts: Counter[tuple[int, int]] = Counter()
    shapes = set()
    scanned = not_raw = 0
    bar = tqdm(files, unit="file")
    for f in bar:
        try:
            hits, shape = scan_file(f, args.threshold)
        except rawpy.LibRawFileUnsupportedError:
            not_raw += 1  # JPEGs, sidecars, etc.: skipped quietly
            continue
        except (rawpy.LibRawError, OSError, ValueError) as e:
            bar.write(f"{f.name}: skipped ({e})", file=sys.stderr)
            continue
        scanned += 1
        shapes.add(shape)
        counts.update(hits)
        bar.set_postfix_str(f"{f.name}: {len(hits)} outliers")

    if not_raw:
        print(f"Ignored {not_raw} non-raw files", file=sys.stderr)
    if not scanned:
        print(f"No usable raw files in {args.folder}", file=sys.stderr)
        return 1
    if len(shapes) > 1:
        # shortcut: coordinates are compared across all files, fine for one camera; split by shape if mixing bodies.
        print(f"Warning: mixed sensor sizes {shapes}, results may be unreliable", file=sys.stderr)

    min_files = max(1, round(args.min_ratio * scanned))
    stuck = sorted(((pos, n) for pos, n in counts.items() if n >= min_files), key=lambda x: -x[1])
    print(f"\n{len(stuck)} suspected stuck pixels (flagged in >= {min_files}/{scanned} files):")
    print("row,col,files")
    for (r, c), n in stuck:
        print(f"{r},{c},{n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
