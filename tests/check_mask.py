#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validate a fast_scrubber mask against its input image, and optionally against
reference masks (another runtime's output, or a stored regression reference).

Usage:
    python tests/check_mask.py --input orig.mgz --mask out/orig_fastscrubber_mask.nii.gz
    python tests/check_mask.py --input orig.mgz --mask docker/... --reference env/... --min_dice 0.999

Exits non-zero, listing every failed check, if the mask:
  - is not on the input's voxel grid (shape and affine),
  - holds values other than 0 and 1,
  - covers a fraction of the volume outside [--min_fraction, --max_fraction],
  - has a Dice below --min_dice with any --reference mask.
"""

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np


def load(path: Path) -> nib.spatialimages.SpatialImage:
    if not path.is_file():
        sys.exit(f"[FAIL] File not found: {path}")
    return nib.load(str(path))


def dice(a: np.ndarray, b: np.ndarray) -> float:
    total = int(a.sum()) + int(b.sum())
    return 1.0 if total == 0 else 2.0 * int(np.logical_and(a, b).sum()) / total


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a fast_scrubber mask.")
    parser.add_argument("--input", required=True, type=Path, help="Image given to fastscrubber.py.")
    parser.add_argument("--mask", required=True, type=Path, help="Predicted mask to check.")
    parser.add_argument("--reference", type=Path, action="append", default=[],
                        help="Mask to compare against (repeatable).")
    parser.add_argument("--min_dice", type=float, default=0.999,
                        help="Minimum Dice with each --reference (default: 0.999).")
    parser.add_argument("--min_fraction", type=float, default=0.01,
                        help="Minimum foreground fraction of the volume (default: 0.01).")
    parser.add_argument("--max_fraction", type=float, default=0.60,
                        help="Maximum foreground fraction of the volume (default: 0.60).")
    parser.add_argument("--json", type=Path, default=None, help="Also write the summary here.")
    args = parser.parse_args()

    image = load(args.input)
    mask  = load(args.mask)
    data  = np.asarray(mask.dataobj)
    fg    = data > 0

    failures = []
    summary = {
        "mask": str(args.mask),
        "shape": [int(n) for n in mask.shape],
        "input_shape": [int(n) for n in image.shape],
        "foreground_voxels": int(fg.sum()),
        "foreground_fraction": float(fg.mean()),
        "volume_ml": float(fg.sum() * np.prod(mask.header.get_zooms()[:3]) / 1000.0),
        "labels": sorted(float(v) for v in np.unique(data)),
        "references": {},
    }

    if mask.shape != image.shape:
        failures.append(f"shape {mask.shape} != input shape {image.shape}")
    if not np.allclose(mask.affine, image.affine, atol=1e-4):
        failures.append(f"affine differs from the input's:\n{mask.affine}\nvs\n{image.affine}")
    if not set(summary["labels"]) <= {0.0, 1.0}:
        failures.append(f"labels {summary['labels']} are not a subset of {{0, 1}}")
    if not args.min_fraction <= summary["foreground_fraction"] <= args.max_fraction:
        failures.append(
            f"foreground fraction {summary['foreground_fraction']:.4f} outside "
            f"[{args.min_fraction}, {args.max_fraction}]"
        )

    for reference in args.reference:
        ref = np.asarray(load(reference).dataobj) > 0
        if ref.shape != fg.shape:
            failures.append(f"reference {reference} has shape {ref.shape}, mask has {fg.shape}")
            continue
        score = dice(fg, ref)
        summary["references"][str(reference)] = {
            "dice": score,
            "differing_voxels": int(np.logical_xor(fg, ref).sum()),
        }
        if score < args.min_dice:
            failures.append(f"Dice {score:.6f} with {reference} < {args.min_dice}")

    summary["status"] = "FAIL" if failures else "PASS"
    summary["failures"] = failures
    text = json.dumps(summary, indent=2)
    print(text)
    if args.json:
        args.json.write_text(text + "\n")

    if failures:
        for failure in failures:
            print(f"[FAIL] {failure}", file=sys.stderr)
        sys.exit(1)
    print(f"[PASS] {args.mask}")


if __name__ == "__main__":
    main()
