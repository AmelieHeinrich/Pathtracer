#!/usr/bin/env python3
"""Compare two PFM captures from the renderer.

The renderer has no tonemapper and no exposure control, so its output is linear radiance and
the differences worth measuring are a few percent against Monte Carlo noise. Neither survives
being quantised to eight bits through a gamma curve, which is why this reads the .pfm that
--capture writes rather than the .png.

    ./build/.../Pathtracer --scene scenes/cornell.pts --sky 0 \\
        --capture before.pfm --samples 512
    # ... change something ...
    tools/compare_images.py before.pfm after.pfm --write-diff diff.png

Captures are bit-reproducible for a fixed scene, sample count and GPU -- the sampler is seeded
from the pixel and the frame index and nothing else -- so "no change" really does mean a
maximum absolute difference of exactly zero, and this script says so rather than hedging.

numpy only; see tools/bake_assets.py for the stdlib-only convention that applies to the asset
baker but not here.
"""

import argparse
import sys

try:
    import numpy as np
except ImportError:
    sys.exit("compare_images: numpy is required -- install it with 'pacman -S python-numpy'")


def read_pfm(path):
    """Reads a colour PFM. Returns a float32 array shaped (height, width, 3), top row first."""
    with open(path, "rb") as file:
        if file.readline().strip() != b"PF":
            raise ValueError(f"{path}: not a colour PFM")

        # The renderer writes a plain "W H" line, but the format allows comments and split
        # tokens, so read tokens rather than assuming one line.
        dimensions = []
        while len(dimensions) < 2:
            line = file.readline()
            if not line:
                raise ValueError(f"{path}: truncated header")
            if line.startswith(b"#"):
                continue
            dimensions.extend(int(token) for token in line.split())
        width, height = dimensions

        scale = float(file.readline().strip())
        data = np.frombuffer(file.read(width * height * 3 * 4), dtype=np.float32)

    if data.size != width * height * 3:
        raise ValueError(f"{path}: expected {width * height * 3} samples, got {data.size}")
    if scale > 0:  # big endian
        data = data.byteswap()

    # PFM rows run bottom to top.
    return data.reshape(height, width, 3)[::-1]


def luminance(image):
    """Rec. 709 luma, matching the primaries the renderer's linear sRGB output is in."""
    return image @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def write_diff(path, before, after, gain):
    """A false-colour absolute difference: blue is small, red is large, scaled by `gain`."""
    try:
        from PIL import Image
    except ImportError:
        print("  (skipping --write-diff: Pillow is not installed)", file=sys.stderr)
        return

    magnitude = np.clip(np.abs(luminance(after) - luminance(before)) * gain, 0.0, 1.0)
    # Blue -> cyan -> yellow -> red, which keeps small differences legible instead of
    # collapsing them into near-black the way a plain magnitude ramp does.
    stops = np.array([[0, 0, 0.4], [0, 0.7, 0.7], [1, 1, 0], [1, 0, 0]], dtype=np.float32)
    position = magnitude * (len(stops) - 1)
    low = np.clip(np.floor(position), 0, len(stops) - 2).astype(int)
    blend = (position - low)[..., None]
    coloured = stops[low] * (1 - blend) + stops[low + 1] * blend

    Image.fromarray((coloured * 255).astype(np.uint8)).save(path)
    print(f"  wrote {path} (gain {gain:g})")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("before")
    parser.add_argument("after")
    parser.add_argument("--write-diff", metavar="OUT.png",
                        help="write a false-colour difference image")
    parser.add_argument("--diff-gain", type=float, default=20.0,
                        help="scale applied to the difference image (default 20)")
    parser.add_argument("--tolerance", type=float, default=0.0,
                        help="exit non-zero if RMSE exceeds this (default 0, exact match)")
    args = parser.parse_args()

    before = read_pfm(args.before)
    after = read_pfm(args.after)
    if before.shape != after.shape:
        sys.exit(f"shape mismatch: {before.shape} vs {after.shape}")

    difference = after.astype(np.float64) - before.astype(np.float64)
    rmse = float(np.sqrt(np.mean(difference**2)))
    max_absolute = float(np.max(np.abs(difference)))

    # Relative to the mean level rather than per pixel: a per-pixel relative error is
    # dominated by near-black pixels where the denominator is noise.
    mean_level = float(np.mean(np.abs(before.astype(np.float64))))
    relative = rmse / mean_level if mean_level > 0 else float("nan")
    changed = int(np.count_nonzero(np.any(difference != 0, axis=-1)))

    print(f"{args.before} -> {args.after}")
    print(f"  resolution      {before.shape[1]}x{before.shape[0]}")
    print(f"  RMSE            {rmse:.6g}")
    print(f"  max abs diff    {max_absolute:.6g}")
    print(f"  RMSE / mean     {relative:.4%}")
    print(f"  pixels changed  {changed} of {before.shape[0] * before.shape[1]}"
          f" ({changed / (before.shape[0] * before.shape[1]):.2%})")

    if max_absolute == 0.0:
        print("  IDENTICAL")
    if args.write_diff:
        write_diff(args.write_diff, before, after, args.diff_gain)

    if rmse > args.tolerance:
        print(f"  FAIL: RMSE {rmse:.6g} exceeds tolerance {args.tolerance:.6g}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
