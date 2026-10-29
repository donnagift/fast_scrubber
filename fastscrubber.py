#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fast_scrubber: nnUNetv2-based brain mask inference tool.

Usage:
    fastscrubber.py INPUT OUTPUT [options]

    fastscrubber.py sub-01/mri/orig.mgz sub-01_fastscrubber_mask.nii.gz
    fastscrubber.py orig.nii.gz mask.mgz --device cpu --num_cpus 8

INPUT:  FastSurfer's conformed mri/orig.mgz, or a NIfTI conversion of it
        (.mgz, .nii.gz or .nii).
OUTPUT: the predicted brain mask (0/1), on the same voxel grid as INPUT; the
        extension picks the format (.nii.gz, .nii or .mgz).

From Python:
    from fastscrubber import scrub
    scrub("orig.mgz", "mask.nii.gz", device="cpu", num_cpus=8)
"""

import argparse
import gzip
import multiprocessing
import os
import shutil
import sys
import tempfile
from pathlib import Path


# --- Internal constants ------------------------------------------------------

SCRIPT_DIR          = Path(__file__).resolve().parent
DEFAULT_WEIGHTS_DIR = SCRIPT_DIR / "weights/fast_scrubber/nnUNet_results"
WEIGHTS_ENV_VAR     = "FASTSCRUBBER_WEIGHTS"

DATASET_NAME = "Dataset500_Segmentation"
MODEL_NAME   = "nnUNetTrainer__nnUNetPlans__3d_fullres"
CHECKPOINT   = "checkpoint_final.pth"
ALL_FOLDS    = ["0", "1", "2", "3", "4"]
DEVICES      = ["auto", "cuda", "cpu", "mps"]

IMAGE_SUFFIXES = (".nii.gz", ".nii", ".mgz")

# Voxel order of FastSurfer's conformed images, which the model was trained on.
TRAINING_ORIENTATION = ("L", "I", "A")

PROBABILITIES_SUFFIX = "_probabilities.npz"


class FastScrubberError(RuntimeError):
    """An input, output, model or device problem, with a message for the user."""


# --- Helpers -----------------------------------------------------------------

def default_num_cpus() -> int:
    """CPUs this process may use minus 2, with a minimum of 1."""
    try:
        available = len(os.sched_getaffinity(0))
    except AttributeError:  # macOS
        available = multiprocessing.cpu_count()
    return max(1, available - 2)


def image_suffix(path: Path) -> str:
    """The image extension of ``path`` (.nii.gz, .nii or .mgz), or ''."""
    name = path.name.lower()
    return next((suffix for suffix in IMAGE_SUFFIXES if name.endswith(suffix)), "")


def limit_threads(num_threads: int) -> None:
    """Set the thread count of the numerical libraries and of nnUNet itself.

    Takes full effect only before numpy/torch/nnunetv2 are imported: OpenBLAS,
    MKL and OpenMP size their pools from these variables when they load, and
    nnunetv2 reads nnUNet_def_n_proc once, as the cap it puts on torch's threads
    during the sliding-window prediction (8 when unset).
    """
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "nnUNet_def_n_proc"):
        os.environ[var] = str(num_threads)


def resolve_model_dir(weights: Path, folds: list) -> Path:
    """Locate the trained model folder and check every file inference needs.

    ``weights`` may be the nnUNet_results directory or the model folder itself.
    """
    model_dir = weights
    if not (model_dir / "plans.json").is_file():
        model_dir = weights / DATASET_NAME / MODEL_NAME

    required = [model_dir / "dataset.json", model_dir / "plans.json"]
    required += [model_dir / f"fold_{fold}" / CHECKPOINT for fold in folds]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FastScrubberError(
            f"Model weights not found under {weights}. Missing:\n  "
            + "\n  ".join(missing)
            + "\nDownload them with `python download_weights.py` "
            "(or from https://osf.io/x95g7), or point --weights / "
            f"${WEIGHTS_ENV_VAR} at an existing nnUNet_results directory."
        )
    return model_dir


def resolve_device(device: str) -> str:
    """The device to run on; 'auto' takes a GPU when one is visible."""
    import torch

    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        print(
            "[WARN] No GPU visible (docker run --gpus all / apptainer run --nv "
            "expose one); running on the CPU.",
            flush=True,
        )
        return "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise FastScrubberError(
            "--device cuda was requested but PyTorch sees no CUDA GPU. "
            "Use --device cpu, or expose the GPU to the container "
            "(docker run --gpus all / apptainer run --nv)."
        )
    if device == "mps" and not torch.backends.mps.is_available():
        raise FastScrubberError("--device mps was requested but MPS is not available.")
    return device


def stage_input(t1w: Path, dest: Path) -> None:
    """Write the input image to ``dest`` as .nii.gz, the model's file ending."""
    import nibabel as nib

    suffix = image_suffix(t1w)
    if suffix == ".nii.gz":
        print(f"[INFO] Copying {t1w} -> {dest}", flush=True)
        shutil.copyfile(t1w, dest)
    elif suffix == ".nii":
        print(f"[INFO] Compressing {t1w} -> {dest}", flush=True)
        with open(t1w, "rb") as src, gzip.open(dest, "wb") as out:
            shutil.copyfileobj(src, out)
    else:
        # Same voxels and vox2ras as `mri_convert orig.mgz out.nii.gz`, which
        # produced the training images; no FreeSurfer install needed.
        import numpy as np

        print(f"[INFO] Converting {t1w} -> {dest}", flush=True)
        img  = nib.load(str(t1w))
        data = np.asarray(img.dataobj)
        nii  = nib.Nifti1Image(data, affine=img.affine, header=img.header)
        nib.save(nii, str(dest))

    orientation = nib.aff2axcodes(nib.load(str(dest)).affine)
    if orientation != TRAINING_ORIENTATION:
        print(
            f"[WARN] Input voxel orientation is {''.join(orientation)}, but the "
            f"model was trained on FastSurfer-conformed images "
            f"({''.join(TRAINING_ORIENTATION)}). Pass FastSurfer's mri/orig.mgz "
            "for reliable predictions.",
            flush=True,
        )


def predict(staged_input: Path, output_truncated: Path, model_dir: Path, *, device: str,
            folds: list, tta: bool, save_probabilities: bool, num_cpus: int) -> None:
    """Run nnUNetv2 inference with the same settings as nnUNetv2_predict.

    nnUNetv2_predict preprocesses and exports in worker processes and hands the
    image and the logits over through shared memory. A 0.5 mm volume does not
    fit in Docker's default 64 MB /dev/shm, so the workers die there. With one
    case the workers save no time, so the same three steps run here in turn.
    """
    import torch
    import nnunetv2.inference.predict_from_raw_data as nnunet_predict
    from nnunetv2.inference.export_prediction import export_prediction_from_logits

    # nnUNetv2_predict would give torch every CPU of the machine; respect
    # num_cpus instead. On a GPU, extra torch threads do not help.
    if device == "cpu":
        torch.set_num_threads(num_cpus)
        # The cap limit_threads sets, also when nnunetv2 was imported before it
        nnunet_predict.default_num_processes = num_cpus
    else:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

    predictor = nnunet_predict.nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=tta,
        perform_everything_on_device=True,
        device=torch.device(device),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=sys.stderr.isatty(),
    )
    predictor.initialize_from_trained_model_folder(
        str(model_dir),
        [int(fold) for fold in folds],
        checkpoint_name=CHECKPOINT,
    )

    # The steps of nnunetv2.inference.data_iterators.preprocess_fromfiles_save_to_queue
    # and predict_from_data_iterator, for a single case.
    print("[INFO] Preprocessing", flush=True)
    preprocessor = predictor.configuration_manager.preprocessor_class(verbose=False)
    data, _, properties = preprocessor.run_case(
        [str(staged_input)],
        None,
        predictor.plans_manager,
        predictor.configuration_manager,
        predictor.dataset_json,
    )
    data = torch.from_numpy(data).to(dtype=torch.float32, memory_format=torch.contiguous_format)

    print(f"[INFO] Running the network on {tuple(data.shape[1:])} voxels", flush=True)
    logits = predictor.predict_logits_from_preprocessed_data(data).cpu()
    del data

    print("[INFO] Resampling to the input grid and exporting", flush=True)
    export_prediction_from_logits(
        logits,
        properties,
        predictor.configuration_manager,
        predictor.plans_manager,
        predictor.dataset_json,
        str(output_truncated),
        save_probabilities,
    )


def write_mask(prediction: Path, input_image: Path, output: Path) -> None:
    """Check the prediction is on the input's grid, then write it to ``output``.

    The file is written next to ``output`` and renamed into place, so a failed
    run never leaves a partial mask behind.
    """
    import nibabel as nib
    import numpy as np

    mask  = nib.load(str(prediction))
    image = nib.load(str(input_image))
    if mask.shape != image.shape:
        raise FastScrubberError(
            f"Predicted mask shape {mask.shape} != input shape {image.shape}."
        )
    data = np.asarray(mask.dataobj).astype(np.uint8)
    voxels = int(np.count_nonzero(data))
    volume_ml = voxels * float(np.prod(mask.header.get_zooms()[:3])) / 1000.0
    print(f"[INFO] Mask: {voxels} voxels, {volume_ml:.1f} mL", flush=True)
    if voxels == 0:
        print("[WARN] The predicted mask is empty.", flush=True)

    suffix  = image_suffix(output)
    partial = prediction.with_name(f"partial{suffix}")
    if suffix == ".nii.gz":
        shutil.copyfile(prediction, partial)
    elif suffix == ".nii":
        nib.save(mask, str(partial))
    else:
        nib.save(nib.MGHImage(data, mask.affine), str(partial))

    staged = output.with_name(f".{output.name}.partial")
    shutil.copyfile(partial, staged)
    os.replace(staged, output)


# --- Main function -----------------------------------------------------------

def scrub(input_image, output, *, device: str = "auto", folds=ALL_FOLDS,
          tta: bool = False, save_probabilities: bool = False,
          num_cpus: int = None, weights=None) -> Path:
    """Predict the FastScrubber brain mask of one image and write it to ``output``.

    Parameters
    ----------
    input_image : path-like
        FastSurfer's conformed mri/orig.mgz, or a NIfTI of it (.mgz, .nii.gz, .nii).
    output : path-like
        Mask to write (.nii.gz, .nii or .mgz); its directory is created. With
        ``save_probabilities``, the softmax maps go next to it, in
        ``<output without extension>_probabilities.npz``.
    device : 'auto', 'cuda', 'cpu' or 'mps'
        'auto' (default) runs on a visible GPU, else on the CPU.
    folds : folds of the 5-fold ensemble to use (default: all).
    tta : also predict on mirrored copies and average them (test-time
        augmentation): up to 8x slower, slightly more accurate. Off by default.
    num_cpus : CPU threads for CPU inference (default: available CPUs - 2).
    weights : nnUNet_results directory, or the model folder itself (default:
        $FASTSCRUBBER_WEIGHTS, else weights/ next to this file).

    Returns the path of the written mask. Raises FastScrubberError for problems
    with the input, output, weights or device. Sets the thread environment
    variables of limit_threads() in this process.
    """
    input_image = Path(input_image).resolve()
    output      = Path(output).resolve()
    folds       = [str(fold) for fold in folds]
    num_cpus    = num_cpus if num_cpus is not None else default_num_cpus()
    weights     = Path(weights or os.environ.get(WEIGHTS_ENV_VAR, DEFAULT_WEIGHTS_DIR)).resolve()

    if not input_image.is_file():
        raise FastScrubberError(f"Input file not found: {input_image}")
    if not image_suffix(input_image):
        raise FastScrubberError(
            f"Unsupported input format: '{input_image.name}'. "
            "Provide a .nii.gz, .nii, or .mgz file."
        )
    if not image_suffix(output):
        raise FastScrubberError(
            f"Unsupported output format: '{output.name}'. "
            "Name the mask .nii.gz, .nii, or .mgz."
        )
    if output == input_image:
        raise FastScrubberError("The output path must differ from the input path.")
    if num_cpus < 1:
        raise FastScrubberError("num_cpus must be at least 1.")
    if not set(folds) <= set(ALL_FOLDS):
        raise FastScrubberError(f"Folds must be among {' '.join(ALL_FOLDS)}, got {' '.join(folds)}.")
    if device not in DEVICES:
        raise FastScrubberError(f"Device must be one of {', '.join(DEVICES)}, got '{device}'.")

    limit_threads(num_cpus)
    model_dir = resolve_model_dir(weights, folds)
    device = resolve_device(device)
    output.parent.mkdir(parents=True, exist_ok=True)

    if device == "cpu":
        print(
            f"[WARN] Running on CPU with {num_cpus} thread(s) -- inference will be "
            "significantly slower than on GPU.",
            flush=True,
        )

    with tempfile.TemporaryDirectory(prefix="fast_scrubber_") as tmpdir:
        tmpdir = Path(tmpdir)
        staged_input  = tmpdir / "input" / "case_0000.nii.gz"
        inference_dir = tmpdir / "inference"
        staged_input.parent.mkdir()
        inference_dir.mkdir()

        # nnunetv2 warns at import time when these are unset; inference itself
        # only uses the explicit model folder.
        os.environ.setdefault("nnUNet_raw", str(tmpdir))
        os.environ.setdefault("nnUNet_preprocessed", str(tmpdir))
        os.environ.setdefault("nnUNet_results", str(weights))

        stage_input(input_image, staged_input)

        print(
            f"[INFO] Predicting with {model_dir} (folds {' '.join(folds)}, "
            f"device {device}, TTA {'on' if tta else 'off'})",
            flush=True,
        )
        predict(staged_input, inference_dir / "case", model_dir, device=device, folds=folds,
                tta=tta, save_probabilities=save_probabilities,
                num_cpus=num_cpus)

        write_mask(inference_dir / "case.nii.gz", staged_input, output)
        print(f"[INFO] Mask written to {output}", flush=True)
        if save_probabilities:
            probabilities = output.with_name(output.name[: -len(image_suffix(output))]
                                             + PROBABILITIES_SUFFIX)
            shutil.copyfile(inference_dir / "case.npz", probabilities)
            print(f"[INFO] Probabilities written to {probabilities}", flush=True)

    return output


# --- Command line ------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="fast_scrubber: predict a cleaned brain mask from FastSurfer's orig.mgz.",
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input image: FastSurfer's mri/orig.mgz (.mgz, .nii.gz, or .nii).",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Output brain mask (.nii.gz, .nii, or .mgz), on the input's voxel grid.",
    )
    parser.add_argument(
        "--device",
        choices=DEVICES,
        default="auto",
        help=(
            "Device for inference: 'auto' (default: a GPU if one is visible, "
            "else the CPU), 'cuda', 'cpu', or 'mps' (Apple Silicon)."
        ),
    )
    parser.add_argument(
        "--num_cpus",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Number of CPU threads for CPU inference. "
            f"Default: available CPUs - 2 (currently {default_num_cpus()} on this machine)."
        ),
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        choices=ALL_FOLDS,
        default=ALL_FOLDS,
        help="Folds to use for prediction (default: all 5).",
    )
    mirroring = parser.add_mutually_exclusive_group()
    mirroring.add_argument(
        "--tta",
        action="store_true",
        default=False,
        help=(
            "Also predict on mirrored copies and average them (test-time augmentation): "
            "up to 8x slower, slightly more accurate. Off by default."
        ),
    )
    mirroring.add_argument(
        "--disable_tta",
        action="store_true",
        help="Accepted for older scripts; TTA is already off unless --tta is given.",
    )
    parser.add_argument(
        "--save_probabilities",
        action="store_true",
        default=False,
        help=f"Also save the softmax maps, to <output without extension>{PROBABILITIES_SUFFIX}.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help=(
            "nnUNet_results directory holding the FastScrubber model "
            f"(default: ${WEIGHTS_ENV_VAR}, else {DEFAULT_WEIGHTS_DIR})."
        ),
    )
    args = parser.parse_args()

    try:
        scrub(
            args.input,
            args.output,
            device=args.device,
            folds=args.folds,
            tta=args.tta,
            save_probabilities=args.save_probabilities,
            num_cpus=args.num_cpus,
            weights=args.weights,
        )
    except FastScrubberError as error:
        sys.exit(f"[ERROR] {error}")

    print("\n[DONE] Inference complete.\n", flush=True)


if __name__ == "__main__":
    main()
