"""Find stuck/hot/dead pixels across a folder of camera raw files.

A defective pixel sits at the same sensor coordinate in every shot, so we flag
outliers per file on the raw Bayer data and keep only coordinates that repeat
across many files. Real image detail (stars, specular highlights) rarely repeats.
"""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import rawpy
from PIL import Image, ImageDraw
from tqdm import tqdm

# Heuristic: below this, chance repeats of real detail can pass the --min-ratio vote.
MIN_RELIABLE_FILES = 10
ZOOM_RADIUS = 16  # crop is (2*16+1)^2 sensor pixels around the defect
ZOOM_SCALE = 12


def find_outliers(raw: np.ndarray, black: float, white: float, threshold: float) -> set[tuple[int, int]]:
    """Return (row, col) of pixels that differ from their same-colour neighbours by > threshold of the range."""
    data = raw.astype(np.int32)
    limit = np.float32(threshold * (white - black))
    hits: set[tuple[int, int]] = set()
    # Each 2x2 Bayer offset is a single-colour plane; compare pixels only to their own colour.
    for dy in (0, 1):
        for dx in (0, 1):
            plane = data[dy::2, dx::2]
            padded = np.pad(plane, 1, mode="edge")
            h, w = plane.shape
            neighbours = [
                padded[1 + oy:1 + oy + h, 1 + ox:1 + ox + w]
                for oy in (-1, 0, 1) for ox in (-1, 0, 1) if (oy, ox) != (0, 0)
            ]
            # The median sits between the min and max neighbour, so a pixel far from the median is
            # also far from one of them: this cheap test keeps every real hit, then the median confirms.
            lo, hi = neighbours[0].copy(), neighbours[0].copy()
            for n in neighbours[1:]:
                np.minimum(lo, n, out=lo)
                np.maximum(hi, n, out=hi)
            rows, cols = np.nonzero((plane - lo > limit) | (hi - plane > limit))
            median = np.median(np.stack([n[rows, cols] for n in neighbours]).astype(np.float32), axis=0)
            far = np.abs(plane[rows, cols].astype(np.float32) - median) > limit
            for r, c in zip(rows[far], cols[far]):
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


def render(path: Path) -> np.ndarray:
    """Demosaic to 8-bit RGB in raw sensor orientation, so array (row, col) == raw_image_visible (row, col)."""
    with rawpy.imread(str(path)) as raw:
        rgb = raw.postprocess(user_flip=0)
        if rgb.shape[:2] != raw.raw_image_visible.shape:
            raise ValueError(f"rendered size {rgb.shape[:2]} != sensor size {raw.raw_image_visible.shape}")
        return rgb


def save_marks(rgb: np.ndarray, pixels: set[tuple[int, int]], base: Path, ext: str) -> None:
    """Write <base>_marked.<ext> (full size, defects circled) and one zoomed crop per defect."""
    h, w = rgb.shape[:2]
    full = Image.fromarray(rgb)
    draw = ImageDraw.Draw(full)
    radius, width = max(15, max(h, w) // 150), max(2, max(h, w) // 1000)
    for r, c in pixels:
        # Black outer + white inner ring stays visible on any colour, including saturated noise.
        draw.ellipse((c - radius - width, r - radius - width, c + radius + width, r + radius + width),
                     outline="black", width=width)
        draw.ellipse((c - radius, r - radius, c + radius, r + radius), outline="white", width=width)
        top, left = max(0, r - ZOOM_RADIUS), max(0, c - ZOOM_RADIUS)
        crop = Image.fromarray(rgb[top:r + ZOOM_RADIUS + 1, left:c + ZOOM_RADIUS + 1])
        # Nearest-neighbour keeps the single defective pixel a crisp square instead of a blur.
        zoom = crop.resize((crop.width * ZOOM_SCALE, crop.height * ZOOM_SCALE), Image.Resampling.NEAREST)
        y, x = (r - top) * ZOOM_SCALE, (c - left) * ZOOM_SCALE
        # Box drawn half a pixel outside the defect so its own colour stays untouched.
        gap = ZOOM_SCALE // 2
        zdraw = ImageDraw.Draw(zoom)
        zdraw.rectangle((x - gap - 2, y - gap - 2, x + ZOOM_SCALE + gap + 1, y + ZOOM_SCALE + gap + 1),
                        outline="black", width=2)
        zdraw.rectangle((x - gap, y - gap, x + ZOOM_SCALE + gap - 1, y + ZOOM_SCALE + gap - 1),
                        outline="white", width=2)
        zoom.save(f"{base}_r{r}_c{c}.{ext}", quality=95)
    full.save(f"{base}_marked.{ext}", quality=95)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("folder", type=Path)
    ap.add_argument("--threshold", type=float, default=0.25,
                    help="min deviation from neighbours as fraction of sensor range (default 0.25)")
    ap.add_argument("--min-ratio", type=float, default=0.5,
                    help="fraction of files a pixel must be flagged in (default 0.5)")
    ap.add_argument("--recursive", action="store_true", help="also scan subfolders")
    ap.add_argument("--mark", type=Path, metavar="DIR",
                    help="write a full-size image per affected file with defects circled, plus zoomed crops")
    ap.add_argument("--mark-format", choices=("png", "jpg"), default="png",
                    help="image format for --mark (default png, lossless)")
    args = ap.parse_args()

    if not args.folder.is_dir():
        ap.error(f"not a folder: {args.folder}")
    pattern = "**/*" if args.recursive else "*"
    # No extension list: LibRaw recognises raw formats by content, so every file is offered to it.
    files = sorted(p for p in args.folder.glob(pattern) if p.is_file() and not p.name.startswith("."))

    counts: Counter[tuple[int, int]] = Counter()
    per_file: dict[Path, set[tuple[int, int]]] = {}
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
        per_file[f] = hits
        bar.set_postfix_str(f"{f.name}: {len(hits)} outliers")

    if not_raw:
        print(f"Ignored {not_raw} non-raw files", file=sys.stderr)
    if not scanned:
        print(f"No usable raw files in {args.folder}", file=sys.stderr)
        return 1
    if len(shapes) > 1:
        # shortcut: coordinates are compared across all files, fine for one camera; split by shape if mixing bodies.
        print(f"Warning: mixed sensor sizes {shapes}, results may be unreliable", file=sys.stderr)

    if scanned < MIN_RELIABLE_FILES:
        print(f"Warning: only {scanned} raw files; with fewer than {MIN_RELIABLE_FILES}, image detail "
              "(stars, highlights) can repeat by chance and be reported", file=sys.stderr)

    min_files = max(1, round(args.min_ratio * scanned))
    stuck = sorted(((pos, n) for pos, n in counts.items() if n >= min_files), key=lambda x: -x[1])
    print(f"{len(stuck)} suspected stuck pixels (flagged in >= {min_files}/{scanned} files)", file=sys.stderr)
    stuck_set = {pos for pos, _ in stuck}
    seen_in: dict[tuple[int, int], list[str]] = {pos: [] for pos in stuck_set}
    for f, hits in per_file.items():
        for pos in hits & stuck_set:
            seen_in[pos].append(f.relative_to(args.folder).as_posix())

    out = csv.writer(sys.stdout, lineterminator="\n")
    out.writerow(["row", "col", "files", "files_pct", "file_names"])
    for (r, c), n in stuck:
        out.writerow([r, c, n, f"{100 * n / scanned:.1f}", ";".join(seen_in[(r, c)])])

    if args.mark:
        affected = {f: hits & stuck_set for f, hits in per_file.items() if hits & stuck_set}
        for f, pixels in tqdm(affected.items(), unit="file", desc="marking"):
            try:
                rgb = render(f)
            except (rawpy.LibRawError, OSError, ValueError) as e:
                tqdm.write(f"{f.name}: not marked ({e})", file=sys.stderr)
                continue
            # One folder per raw, named with its extension and subfolders, so IMG_1.CR3 and IMG_1.NEF never collide.
            rel = f.relative_to(args.folder)
            (args.mark / rel).mkdir(parents=True, exist_ok=True)
            save_marks(rgb, pixels, args.mark / rel / rel.stem, args.mark_format)
    return 0


if __name__ == "__main__":
    sys.exit(main())
