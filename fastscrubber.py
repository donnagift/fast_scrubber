#!/usr/bin/env python3
"""
fast_scrubber: nnUNetv2-based segmentation inference tool.

Usage:
    python fastscrubber.py --t1w /path/to/image.nii.gz --out /path/to/output/
    python fastscrubber.py --t1w /path/to/image.mgz    --out /path/to/output/ --device cpu

Supported inputs: .nii.gz, .nii, .mgz
"""

import argparse
import multiprocessing
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np


# --- Internal constants ------------------------------------------------------

SCRIPT_DIR  = Path(__file__).parent.resolve()
WEIGHTS_DIR = SCRIPT_DIR / "weights"

DATASET_ID   = "500"
DATASET_NAME = "Dataset500_Segmentation"
CONFIG       = "3d_fullres"

CASE_ID               = "input_case"
NNUNET_INPUT_FILENAME = f"{CASE_ID}_0000.nii.gz"


# --- Helpers -----------------------------------------------------------------

def run(cmd: list, env: dict) -> None:
    """Run a subprocess, streaming output, and exit on failure."""
    print(f"\n[CMD] {' '.join(cmd)}\n", flush=True)
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        sys.exit(f"[ERROR] Command failed with exit code {result.returncode}.")


def convert_mgz_to_nii(mgz_path: Path, out_path: Path) -> None:
    """Convert .mgz to .nii.gz using nibabel (no FreeSurfer required)."""
    print(f"[INFO] Converting {mgz_path} -> {out_path}", flush=True)
    img  = nib.load(str(mgz_path))
    data = np.asarray(img.dataobj)
    nii  = nib.Nifti1Image(data, affine=img.affine, header=img.header)
    nib.save(nii, str(out_path))


def resolve_input(t1w: Path, images_dir: Path, env: dict) -> None:
    """Copy or convert the input image into the nnUNet imagesTs directory."""
    dest   = images_dir / NNUNET_INPUT_FILENAME
    suffix = "".join(t1w.suffixes)

    if suffix in (".nii.gz", ".nii"):
        print(f"[INFO] Copying {t1w} -> {dest}", flush=True)
        shutil.copy2(t1w, dest)
    elif suffix == ".mgz":
        convert_mgz_to_nii(t1w, dest)
    else:
        sys.exit(
            f"[ERROR] Unsupported input format: '{suffix}'. "
            "Provide a .nii.gz, .nii, or .mgz file."
        )


def default_num_cpus() -> int:
    """Total CPU count minus 2, with a minimum of 1."""
    return max(1, multiprocessing.cpu_count() - 2)


# --- Main --------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="fast_scrubber: run nnUNetv2 segmentation inference."
    )
    parser.add_argument(
        "--t1w",
        required=True,
        type=Path,
        help="Input T1w image (.nii.gz, .nii, or .mgz).",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output directory where the predicted mask will be saved.",
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        default=["0", "1", "2", "3", "4"],
        help="Folds to use for prediction (default: all 5).",
    )
    parser.add_argument(
        "--save_probabilities",
        action="store_true",
        default=False,
        help="Also save softmax probability maps.",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu", "mps"],
        default="cuda",
        help=(
            "Device for inference: 'cuda' (GPU, default), "
            "'cpu' (no GPU required, slower), 'mps' (Apple Silicon)."
        ),
    )
    parser.add_argument(
        "--num_cpus",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Number of CPU workers for preprocessing and export "
            "(only used when --device cpu). "
            f"Default: total CPUs - 2 (currently {default_num_cpus()} on this machine)."
        ),
    )
    args = parser.parse_args()

    # Resolve num_cpus: only meaningful for CPU inference
    if args.device == "cpu":
        num_cpus = args.num_cpus if args.num_cpus is not None else default_num_cpus()
        if num_cpus < 1:
            sys.exit("[ERROR] --num_cpus must be at least 1.")
    else:
        if args.num_cpus is not None:
            print("[WARN] --num_cpus is ignored when --device is not 'cpu'.", flush=True)
        num_cpus = 1  # not used, but set for clarity

    t1w: Path     = args.t1w.resolve()
    out_dir: Path = args.out.resolve()

    if not t1w.exists():
        sys.exit(f"[ERROR] Input file not found: {t1w}")

    if not WEIGHTS_DIR.exists():
        sys.exit(
            f"[ERROR] Weights directory not found: {WEIGHTS_DIR}\n"
            "Please download weights from https://osf.io/x95g7/files/osfstorage "
            "and place them under weights/ (see README)."
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cpu":
        print(
            f"[WARN] Running on CPU with {num_cpus} worker(s) -- inference will be "
            "significantly slower than on GPU (expect several minutes per case).",
            flush=True,
        )

    with tempfile.TemporaryDirectory(prefix="fast_scrubber_") as tmpdir:
        tmpdir = Path(tmpdir)

        nnunet_raw          = tmpdir / "nnUNet_raw"
        nnunet_preprocessed = tmpdir / "nnUNet_preprocessed"
        images_ts           = nnunet_raw / DATASET_NAME / "imagesTs"
        inference_dir       = tmpdir / "inference"

        images_ts.mkdir(parents=True)
        nnunet_preprocessed.mkdir(parents=True)
        inference_dir.mkdir(parents=True)

        env = os.environ.copy()
        env["nnUNet_raw"]          = str(nnunet_raw)
        env["nnUNet_preprocessed"] = str(nnunet_preprocessed)
        env["nnUNet_results"]      = str(WEIGHTS_DIR)

        resolve_input(t1w, images_ts, env)

        cmd = [
            "nnUNetv2_predict",
            "-i",      str(images_ts),
            "-o",      str(inference_dir),
            "-d",      DATASET_ID,
            "-c",      CONFIG,
            "-f",      *args.folds,
            "-device", args.device,
        ]

        if args.device == "cpu":
            cmd += ["-npp", str(num_cpus), "-nps", str(num_cpus)]

        if args.save_probabilities:
            cmd.append("--save_probabilities")

        run(cmd, env)

        print(f"\n[INFO] Copying results to {out_dir}", flush=True)
        for result_file in inference_dir.iterdir():
            dest_file = out_dir / result_file.name
            shutil.copy2(result_file, dest_file)
            print(f"  -> {dest_file}")

    print("\n[DONE] Inference complete.\n", flush=True)


if __name__ == "__main__":
    main()