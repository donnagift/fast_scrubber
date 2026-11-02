#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a small FastSurfer-like test image from a public BIDS T1w, for CI runners
that cannot reach the lab's BIDS_CI dataset.

Usage:
    python tests/make_fixture.py --out fixture/      # writes fixture/orig.mgz

Downloads sub-01_T1w.nii.gz from OpenNeuro ds000001 (CC0), conforms it the way
`mri_convert --conform` does (1 mm isotropic, LIA, uint8), and crops a block
around the brain's centre so that 0.5 mm inference covers only a few nnUNet
patches (about a minute per fold on 4 CPUs). It exercises the .mgz path,
geometry handling and the model end to end; it is not a quality benchmark.
"""

import argparse
import urllib.request
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import conform


SOURCE_URL = "https://s3.amazonaws.com/openneuro.org/ds000001/sub-01/anat/sub-01_T1w.nii.gz"

# Crop size in 1 mm voxels (LIA axes: left-right, inferior-superior,
# anterior-posterior); 2x2x2 patches of 160x160x96 at 0.5 mm.
CROP = (96, 96, 72)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the CI test image.")
    parser.add_argument("--out", type=Path, required=True, help="Output directory.")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    source = args.out / "sub-01_T1w.nii.gz"
    if not source.is_file():
        print(f"[INFO] Downloading {SOURCE_URL}", flush=True)
        urllib.request.urlretrieve(SOURCE_URL, source)

    img = conform(nib.load(str(source)), out_shape=(256, 256, 256),
                  voxel_size=(1.0, 1.0, 1.0), order=1, orientation="LIA")
    data = np.asarray(img.dataobj, dtype=np.float32)
    data = np.clip(data / np.percentile(data[data > 0], 99.9), 0, 1) * 255

    # Centre the crop on the intensity-weighted centre of the head
    head = data > np.percentile(data, 75)
    centre = np.round(np.argwhere(head).mean(axis=0)).astype(int)
    start = [int(np.clip(c - s // 2, 0, n - s)) for c, s, n in zip(centre, CROP, data.shape)]
    window = tuple(slice(b, b + s) for b, s in zip(start, CROP))

    cropped = img.slicer[window]
    orig = nib.MGHImage(data[window].round().astype(np.uint8), cropped.affine)
    dest = args.out / "orig.mgz"
    nib.save(orig, str(dest))
    print(f"[DONE] {dest}: shape {orig.shape}, "
          f"orientation {''.join(nib.aff2axcodes(orig.affine))}", flush=True)


if __name__ == "__main__":
    main()
