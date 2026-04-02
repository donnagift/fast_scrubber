#---------------- Environment ----------------#

#converts mgz to nii, for image prediction
# Set freesurfer 7 default environment
PATH=$(IFS=':';p=($PATH);unset IFS;p=(${p[@]%%*Freesurfer*});IFS=':';echo "${p[*]}";unset IFS)
unset FSFAST_HOME SUBJECTS_DIR MNI_DIR
export FREESURFER_HOME=/data_/mica1/01_programs/freesurfer-7.3.2 && export MNI_DIR=${FREESURFER_HOME}/mni && source ${FREESURFER_HOME}/FreeSurferEnv.sh
export PATH=${FREESURFER_HOME}/bin/:${FSLDIR}:${FSL_BIN}:${PATH}

# ---------------- Paths ----------------
fastsurfer_dir=/path/to/fastsurfer2.4.2/directory/
imagesTs=/path/to/imagesTs/to/predict

# ---------------- Settings ----------------
session="ses-01"   # <<< change session here

subjects=("sub-HC001" "sub-HC0002")
echo "Starting conversion..."
echo "Session: $session"
echo "Output directory: $imagesTs"
echo "----------------------------------"

# ---------------- Processing ----------------
for subj in "${subjects[@]}"; do

    input="${fastsurfer_dir}/${subj}_${session}/mri/orig.mgz"
    output="${imagesTs}/${subj}_${session}_0000.nii.gz"

    if [ -f "$input" ]; then
        echo "Processing $subj ($session)..."
        mri_convert "$input" "$output"
        echo "Saved → $output"
    else
        echo "WARNING: $input not found, skipping $subj"
    fi

done

echo "----------------------------------"
echo "Done."

