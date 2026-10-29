#!/bin/bash
#
# BIDS-CI test for fast_scrubber: runs fastscrubber.py from the conda
# environment, the Docker image and the Apptainer/Singularity image on the
# lab's BIDS_CI dataset, validates every mask and checks the runtimes agree.
#
# For each case, the BIDS T1w is conformed with FastSurfer 2.4.2's own
# conform.py (`--vox_size min`), which reproduces FastSurfer's mri/orig.mgz
# voxel for voxel, so no full FastSurfer run is needed.
#
# Configure with environment variables; any runtime left empty is skipped:
#   FASTSCRUBBER_PYTHON   python of an env built from environment.yml
#   FASTSCRUBBER_WEIGHTS  nnUNet_results directory for that env
#   FASTSCRUBBER_DOCKER   Docker image, e.g. fast_scrubber:ci
#   FASTSCRUBBER_SIF      Apptainer/Singularity image, e.g. fast_scrubber.sif
#   MODE                  fast (fold 0, no TTA; default) or full (5 folds + TTA)
#   DEVICE                cpu (default) or cuda
#   THREADS               CPU threads per run (default 16)
#   OUTDIR                where outputs go (default /tmp/fast_scrubber_bids_ci_<timestamp>)
#   REFERENCE_DIR         optional dir of <sub>_<ses>_fastscrubber_mask.nii.gz to regress against
#   UPDATE_REFERENCE=1    copy this run's masks into REFERENCE_DIR instead
#
# Example (lab CI runner):
#   FASTSCRUBBER_PYTHON=/path/env/bin/python FASTSCRUBBER_WEIGHTS=/path/weights/fast_scrubber/nnUNet_results \
#   FASTSCRUBBER_DOCKER=fast_scrubber:ci FASTSCRUBBER_SIF=/path/fast_scrubber.sif ./tests/bids_ci_test.sh

set -euo pipefail

bids=${BIDS_CI_RAWDATA:-/data/mica3/BIDS_CI/rawdata}
fastsurfer_img=${FASTSURFER_IMG:-/data_/mica1/01_programs/fastsurfer/fastsurfer-cpu-v2.4.2.sif}
python_env=${FASTSCRUBBER_PYTHON:-}
weights=${FASTSCRUBBER_WEIGHTS:-}
docker_img=${FASTSCRUBBER_DOCKER:-}
sif_img=${FASTSCRUBBER_SIF:-}
mode=${MODE:-fast}
device=${DEVICE:-cpu}
threads=${THREADS:-16}
outdir=${OUTDIR:-/tmp/fast_scrubber_bids_ci_$(date +%Y%m%d_%H%M%S)}
reference_dir=${REFERENCE_DIR:-}
cross_min_dice=${CROSS_MIN_DICE:-0.999}
reference_min_dice=${REFERENCE_MIN_DICE:-0.99}

# <subject> <session> <T1w relative to the session directory>
cases=${CASES:-"sub-mri3T:ses-01:anat/sub-mri3T_ses-01_run-1_T1w.nii.gz sub-mri7T:ses-01:anat/sub-mri7T_ses-01_UNIT1.nii.gz"}

repo=$(cd "$(dirname "$0")/.." && pwd)
container_cmd=$(command -v apptainer || command -v singularity || true)

# Inference flags
case "${mode}" in
  fast) flags=(--folds 0) ;;
  full) flags=(--tta) ;;
  *) echo "[ERROR] MODE must be 'fast' or 'full', got '${mode}'"; exit 1 ;;
esac
flags+=(--device "${device}")
if [[ "${device}" == "cpu" ]]; then flags+=(--num_cpus "${threads}"); fi
docker_gpu=(); sif_gpu=()
if [[ "${device}" == "cuda" ]]; then docker_gpu=(--gpus all); sif_gpu=(--nv); fi

runtimes=()
if [[ -n "${python_env}" ]]; then runtimes+=(env); fi
if [[ -n "${docker_img}" ]]; then runtimes+=(docker); fi
if [[ -n "${sif_img}" ]]; then runtimes+=(apptainer); fi
if [[ ${#runtimes[@]} -eq 0 ]]; then
  echo "[ERROR] Set at least one of FASTSCRUBBER_PYTHON, FASTSCRUBBER_DOCKER, FASTSCRUBBER_SIF"
  exit 1
fi
if [[ ! -f "${fastsurfer_img}" ]]; then
  echo "[ERROR] FastSurfer image not found: ${fastsurfer_img}"; exit 1
fi
if [[ -z "${container_cmd}" ]]; then
  echo "[ERROR] apptainer or singularity is required to run conform.py from ${fastsurfer_img}"; exit 1
fi

mkdir -p "${outdir}"
outdir=$(cd "${outdir}" && pwd)
echo "BIDS-CI fast_scrubber test"
echo "  BIDS:     ${bids}"
echo "  Output:   ${outdir}"
echo "  Runtimes: ${runtimes[*]}"
echo "  Mode:     ${mode} (${flags[*]})"

# Python to run tests/check_mask.py with: any runtime that has nibabel
check_mask() {
  if [[ -n "${python_env}" ]]; then
    "${python_env}" -s "${repo}/tests/check_mask.py" "$@"
  elif [[ -n "${sif_img}" ]]; then
    "${container_cmd}" exec --cleanenv -B "${outdir}" -B "${repo}" "${sif_img}" \
      python "${repo}/tests/check_mask.py" "$@"
  else
    docker run --rm --user "$(id -u):$(id -g)" -v "${outdir}:${outdir}" -v "${repo}:${repo}:ro" \
      --entrypoint python "${docker_img}" "${repo}/tests/check_mask.py" "$@"
  fi
}

run_fastscrubber() {
  local runtime=$1 t1w=$2 mask=$3
  case "${runtime}" in
    env)
      # -s: packages in ~/.local must not shadow the env's own
      FASTSCRUBBER_WEIGHTS="${weights}" "${python_env}" -s "${repo}/fastscrubber.py" \
        "${t1w}" "${mask}" "${flags[@]}" ;;
    docker)
      docker run --rm "${docker_gpu[@]}" --user "$(id -u):$(id -g)" -v "${outdir}:${outdir}" \
        "${docker_img}" "${t1w}" "${mask}" "${flags[@]}" ;;
    apptainer)
      "${container_cmd}" run --cleanenv "${sif_gpu[@]}" -B "${outdir}" \
        "${sif_img}" "${t1w}" "${mask}" "${flags[@]}" ;;
  esac
}

results=()
status=0
for entry in ${cases}; do
  IFS=: read -r sub ses rel <<< "${entry}"
  t1w_bids="${bids}/${sub}/${ses}/${rel}"
  case_id="${sub}_${ses}"
  case_dir="${outdir}/${case_id}"
  mkdir -p "${case_dir}"
  echo -e "\n=== ${case_id}: ${t1w_bids}"

  if [[ ! -f "${t1w_bids}" ]]; then
    echo "[FAIL] BIDS input not found: ${t1w_bids}"
    results+=("${case_id} input FAIL"); status=1; continue
  fi

  # FastSurfer's orig.mgz from the BIDS T1w
  orig="${case_dir}/${case_id}_orig.mgz"
  "${container_cmd}" exec --cleanenv -B "${bids}:${bids}:ro" -B "${case_dir}" "${fastsurfer_img}" \
    python3 /fastsurfer/FastSurferCNN/data_loader/conform.py \
    -i "${t1w_bids}" -o "${orig}" --vox_size min

  baseline=""
  for runtime in "${runtimes[@]}"; do
    out="${case_dir}/${runtime}"
    mask="${out}/${case_id}_fastscrubber_mask.nii.gz"
    echo -e "\n--- ${case_id} / ${runtime}"
    start=$(date +%s)
    if ! run_fastscrubber "${runtime}" "${orig}" "${mask}"; then
      echo "[FAIL] fastscrubber.py exited non-zero (${runtime})"
      results+=("${case_id} ${runtime} FAIL(run)"); status=1; continue
    fi
    seconds=$(( $(date +%s) - start ))

    refs=()
    if [[ -n "${baseline}" ]]; then refs+=(--reference "${baseline}"); fi
    if check_mask --input "${orig}" --mask "${mask}" "${refs[@]}" --min_dice "${cross_min_dice}" \
         --json "${out}/check.json"; then
      results+=("${case_id} ${runtime} PASS ${seconds}s")
    else
      results+=("${case_id} ${runtime} FAIL(check) ${seconds}s"); status=1
    fi
    if [[ -z "${baseline}" && -f "${mask}" ]]; then baseline="${mask}"; fi
  done

  # Regression against stored reference masks
  if [[ -n "${reference_dir}" && -n "${baseline}" ]]; then
    reference="${reference_dir}/${case_id}_fastscrubber_mask.nii.gz"
    if [[ "${UPDATE_REFERENCE:-0}" == "1" ]]; then
      mkdir -p "${reference_dir}" && cp "${baseline}" "${reference}"
      echo "[INFO] Reference updated: ${reference}"
    elif [[ -f "${reference}" ]]; then
      if check_mask --input "${orig}" --mask "${baseline}" --reference "${reference}" \
           --min_dice "${reference_min_dice}" --json "${case_dir}/reference_check.json"; then
        results+=("${case_id} reference PASS")
      else
        results+=("${case_id} reference FAIL"); status=1
      fi
    else
      echo "[WARN] No reference mask at ${reference} (set UPDATE_REFERENCE=1 to create it)"
    fi
  fi
done

echo -e "\n=== Summary (${outdir})"
printf '  %s\n' "${results[@]}"
if [[ ${status} -eq 0 ]]; then echo "BIDS-CI: PASS"; else echo "BIDS-CI: FAIL"; fi
exit ${status}
