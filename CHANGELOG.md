# Changelog

All notable changes to fast_scrubber are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and version numbers
follow [Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-11-13

First tagged release. fast_scrubber is now a standalone tool: one command takes
FastSurfer's `orig.mgz` and writes a brain mask. It runs from a conda
environment, a Docker image or an Apptainer/Singularity image.

The model is unchanged: nnU-Net 2.5.1 `3d_fullres`, Dataset500, 5 folds,
trained on 200 expert-reviewed 7T MP2RAGE scans (129 subjects). The weights are
on OSF: https://osf.io/x95g7.

### Added

- `fastscrubber.py INPUT OUTPUT`, a command-line tool for inference.
  - It reads `.mgz`, `.nii.gz` or `.nii`, so FreeSurfer and `mri_convert` are not needed.
  - It writes a 0/1 mask on the input's voxel grid. The output extension picks the format: `.nii.gz`, `.nii` or `.mgz`.
  - Options: `--device auto|cuda|cpu|mps`, `--num_cpus`, `--folds`, `--tta`,
    `--save_probabilities` and `--weights` (or the `FASTSCRUBBER_WEIGHTS` environment variable).
  - It warns when the input is not in FastSurfer's conformed LIA orientation.
- `scrub()`, the same inference as a Python function (`from fastscrubber import scrub`).
- `download_weights.py`, which fetches the model from OSF and checks each file against a pinned SHA-256.
- A `Dockerfile` with the weights built in. It runs on the CPU, or on a GPU with `--gpus all`.
  An Apptainer/Singularity image can be built from it, and runs on a GPU with `--nv`.
- CI:
  - `.github/workflows/ci.yml` builds the environment, the Docker image and the Apptainer image.
    It runs each on a small public OpenNeuro T1w and checks every mask.
  - `.github/workflows/bids-ci.yml` runs `tests/bids_ci_test.sh` on the lab's BIDS_CI dataset.
- `CHANGELOG.md`, a version in `CITATION.cff`, and `mailmap.txt`.

### Changed

- Test-time augmentation (mirroring) is now off by default. Turn it on with `--tta`.
  - All 5 folds still run.
  - The default is up to 8x faster: 64-68 min instead of ~9 h for a 0.8 mm 3T image on 8 CPU threads,
    and 2.4 min instead of 13.5 min for a 0.5 mm 7T image on a GPU.
  - Accuracy barely changes. On 200 7T cross-validation cases, Dice against the manual masks went from 0.9909 to 0.9906.
    On 10 held-out 7T sessions, masks with and without TTA agreed at Dice 0.9988-0.9992.
  - `--disable_tta` is still accepted, so older scripts keep working. It changes nothing.
- Inference runs in a single process, without nnU-Net's worker processes.
  Those workers died in Docker's default 64 MB `/dev/shm` on full-size images.
- `--num_cpus` now sets nnU-Net's thread cap too (`nnUNet_def_n_proc`).
  Before, nnU-Net capped PyTorch at 8 threads whatever was requested.
- `fastscrubber_env.yml` replaces `environment.yml` and `requirements.txt`.
  - It uses conda-forge only, so there is no Anaconda Terms-of-Service prompt.
  - It pins Python 3.9.20, PyTorch 2.4.1 (CUDA 12.4 wheels) and nnU-Net 2.5.1, plus the other
    packages at the versions the lab runs in production.
  - It sets `PYTHONNOUSERSITE=1`, so packages in `~/.local` cannot shadow the environment's own.
- The README now covers the three ways to install, how to use the tool, runtimes and the tests.

### Fixed

- The conda environment did not build. With `dicom2nifti` unpinned, pip picked a `python-gdcm`
  that has no Python 3.9 wheel.
- The Docker image did not build: the environment name was wrong and the weights path lacked `fast_scrubber/`.
  The image no longer installs a CUDA toolkit it does not need, because the PyTorch wheels ship the CUDA runtime.
- `fastscrubber.py` no longer needs an activated conda environment to find its tools.

### Validation

- On the CPU at 8 threads, fold-0 masks and probabilities are bit-identical to the lab's production
  `nnUNetv2_predict` output.
- On a GPU, the full 5-fold ensemble matches the production mask at Dice 0.999977.
- The BIDS_CI test passes on `sub-mri3T` and `sub-mri7T` from the environment, Docker and Apptainer.
- `ci.yml` passed on GitHub on the merge of #1.

### Known issues

- Inputs must be FastSurfer-conformed `orig.mgz` images, or NIfTI conversions of them.
  Raw scanner T1w images are not supported.
- `scripts/run_fastscrubber.sh` expects FastSurfer 2.4.2.
  In FastSurfer 2.5.4 and 2.6.0, `aseg.auto.mgz` is not cut to the brain mask.
  White matter labelled outside the mask can then make `mri_normalize` fail, and the surfaces collapse.
  With those versions, mask `aseg.auto.mgz` with `mask.mgz` before running the surface stage.
- No image is published on Docker Hub yet. Build it from the `Dockerfile`.

[1.0.0]: https://github.com/donnagift/fast_scrubber/releases/tag/v1.0.0
