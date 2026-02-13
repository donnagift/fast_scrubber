#!/bin/bash


# Script for apply fast_scrubber masks
#Author: Donna Gift Cabalo 02/02/2026
#For fast_scrubber container 
# ./run_fastscrubber.sh sub-PNE048 ses-a1 OUTDIR INF_DIR

export MICAPIPE=/data/mica1/01_programs/micapipe-v0.2.0/
source ${MICAPIPE}/functions/init.sh

APPLYCNNMASK="/path/to/applyCNNmask.sh"

sub=$1
ses=$2
outdir=$3
inf_dir=$4

if [[ -z "$sub" || -z "$ses" || -z "$outdir" || -z "$inf_dir" ]]; then
  echo "Usage: $0 <SUBJECT> <SESSION> <OUTDIR> <INF_DIR>"
  exit 1
fi

echo "Running for subject: $sub session: $ses"
echo "Output directory: $outdir"
echo "Inference directory: $inf_dir"

# Paths
mask_inference="${inf_dir}/${sub}_${ses}.nii.gz"
fsdir="${outdir}/${sub}_${ses}"

# ===== PRE-CHECK BLOCK =====
if [[ ! -f "$mask_inference" ]]; then
  echo "ERROR: Mask not found: $mask_inference"
  exit 1
fi

if [[ ! -d "$fsdir" ]]; then
  echo "ERROR: FastSurfer directory not found: $fsdir"
  exit 1
fi

if [[ ! -d "$outdir" ]]; then
  echo "ERROR: Output directory not found: $outdir"
  exit 1
fi

if [[ ! -d "$inf_dir" ]]; then
  echo "ERROR: Inference directory not found: $inf_dir"
  exit 1
fi

echo "Found mask and FastSurfer directory. Proceeding..."

# ==========================
# 2. Remove old mask and norm
echo "Removing old mask and norm..."
rm -f ${fsdir}/mri/mask.mgz ${fsdir}/mri/norm.mgz

# 3. Replace mask (guarded)
echo "Converting mask to mgz..."
if ! mri_convert "$mask_inference" ${fsdir}/mri/mask.mgz; then
  echo "ERROR: mri_convert failed. Aborting before applyCNNmask."
  exit 1
fi

# 4. Multiply orig_nu with inference mask (guarded)
echo "Creating new norm image..."
if ! mrconvert ${fsdir}/mri/orig_nu.mgz ${fsdir}/mri/orig_nu.nii.gz; then
  echo "ERROR: mrconvert failed. Aborting before applyCNNmask."
  exit 1
fi

if ! fslmaths "$mask_inference" -mul ${fsdir}/mri/orig_nu.nii.gz ${fsdir}/mri/norm.nii.gz; then
  echo "ERROR: fslmaths failed. Aborting before applyCNNmask."
  exit 1
fi

# 5. Convert norm to mgz (guarded)
echo "Converting norm to mgz..."
if ! mrconvert ${fsdir}/mri/norm.nii.gz ${fsdir}/mri/norm.mgz; then
  echo "ERROR: mrconvert (norm) failed. Aborting before applyCNNmask."
  exit 1
fi

# 6. Remove files from previous recon-surf
echo "Cleaning previous recon-surf outputs..."
rm -f ${fsdir}/mri/wm.mgz \
      ${fsdir}/mri/aparc.DKTatlas+aseg.orig.mgz \
      ${fsdir}/mri/orig_nu.nii.gz

# 7. Apply CNN mask (only runs if above succeeded)
echo "Running applyCNNmask.sh ..."
bash ${APPLYCNNMASK} -sub ${sub} -ses ${ses} \
  -out ${outdir} \
  -threads 10

echo "Done for ${sub}_${ses}"

