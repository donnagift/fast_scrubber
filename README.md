<p align="center">
  <img src="fast_scrubber_logo_color.png" width="180">
</p>

<h1 align="center" style="font-size: 3rem;"><em>fast_scrubber</em></h1>

<h2 align="center">
  Automated surface QC. Goodbye to all-day manual QC!
</h2>

# fast_scrubber: Improving brain MRI segmentation at 3T and 7T

## The problem

Ultra-high-field MRI preprocessing remains challenging. Standard pipelines can have difficulty accurately segmenting meninges, blood vessels, and adjacent non-brain tissues, leading to systematic errors and extensive manual quality control (QC) of cortical surfaces.
In our 7T dataset, surfaces were generated using FastSurfer, and we observed typical inclusion of meninges, vessels, tegmentum, and sinuses. These regions must be manually corrected and the surfaces reprocessed, a step that in our experience can take up to a full day per subject.

## The proposed solution

Our approach is to train a dedicated neural network to improve automated segementation and reduce the burden of manual corrections.This enables:
        - improved segmentation quality in challenging regions
        - reduced manual QC time
        - a more robust preprocessing step that integrates into existing pipelines
  
## About the repository

This repository contains the implementation of nnUNet used to train a model for accurately predicting Fastsurfer-based brain masks. 

## How the model was trained


The model was trained with nnunetv2 2.5.1 on manually quality-controlled surfaces extracted from 7T-MP2RAGE UNI images (0.5 × 0.5 × 0.5 mm³) using Fastsurfer v2.4.2. All brain masks were carefully reviewed by expert raters to ensure high precision and reliability. 
The training set consisted of 129 unique subjects (200 total scans, including multiple sessions), comprising healthy controls and participants with epilepsy or autism, with data drawn in part from the MICA-PNI open-access dataset available in OSF (https://osf.io/mhq3f).
Data were split at the subject level, with an 80/20 train–validation ratio, consistent with the standard nnU-Net 5-fold cross-validation procedure.

We provide these training weights ready to use to predict brain masks, automatically cleaned, hence Fastscrubber, to extract more accurate surfaces.

## Installation


## Usage
Prerequisite:
FastSurfer v2.4.2 must be run before running this pipeline.See the FastSurfer documentation for installation and usage: https://deep-mi.org/research/fastsurfer/ 

1. Convert images to predict and save them to `imagesTs` folder
   You can use the `convert_mgz_to_nii.sh` script. Before running it, update the paths in the script:

```bash
 fastsurfer_dir=/path/to/fastsurfer_directory/
 imagesTs=/path/to/fastscrubber/nnunet_raw/imagesTs/
```

2. Predict masks using fast_scrubber

3. Apply the fast_scrubber generated mask using `run_fastscrubber.sh`. This scripts internally calls FastSurfer v2.4.2 `recon-surf.sh`, so FastSurfer must already be installed and configured. 
   
```bash
./run_fastscrubber.sh sub-<SUBJECT_ID> ses-<SESSION_ID> \
/path/to/fastsurfer_directory/ \
/path/to/fast_scrubber_7T/nnUNet_results/Dataset500_Segmentation/nnUNetTrainer__nnUNetPlans__3d_fullres/inference
```
## Some example outputs
<p align="center">
  <img src="https://github.com/donnagift/fast_scrubber/blob/main/images/fast_scrubber_applications.svg" width="600">
</p>


## To train your own model
For detailed instructions on training a custom model, please consult the nnUNet documentation. In summary, you will need to: 

#### 1. Create a conda environment for the nnUNet and follow the installation instructions here: 
> https://github.com/MIC-DKFZ/nnUNet

#### 2. Sort your data and paths according to: 
[setting_up_paths.md](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/setting_up_paths.md)
[dataset_format.md](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md)
[set_environment_variables.md](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/set_environment_variables.md)

#### 3. Copy training images & labels, and test images from Fastsurfer outputs to new directory

Convert `mgz` files to `nii` files before copying

| **Category**   | **Description**                               | **Data Type**         |
|----------------|-----------------------------------------------|-----------------------|
| `imagesTr`     | `orig.mgz` ground truth images for training   | Training images       |
| `labelsTr`     | `orig.mgz`  binarized ground truth images     | Ground truth labels   |
| `imagesTs`     | `orig.mgz` - images to predict                | Testing images        |


## Run the model

### Check data
```bash
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity >> e.g. nnUNetv2_plan_and_preprocess -d 500 --verify_dataset_integrity
```

### Train data
```bash
nnUNetv2_train DATASET_NAME_OR_ID UNET_CONFIGURATION FOLD --npz 
```

> Example:

```bash
TORCHDYNAMO_DISABLE=1 OMP_NUM_THREADS=1 nnUNetv2_train 500 2d 0 --npz   
```
> or

```bash
TORCHDYNAMO_DISABLE=1 OMP_NUM_THREADS=1 nnUNetv2_train 500 3d_fullres 0 --npz –device cuda
```

> run it for 5 folds (0-4), and with different configurations (2d, 3d_lowres, 3d_fullres)- 0

### choose best model

```bash
nnUNetv2_find_best_configuration DATASET_NAME_OR_ID -c CONFIGURATIONS 
```

> Example:

```bash
nnUNetv2_find_best_configuration 500 -c 2d 3d_lowres 3_fullres
```

### Run 2D inference

```bash
# Run 2D inference
nnUNetv2_predict \
-i INPUT_FOLDER \  # Path to the folder containing the input images
-o OUTPUT_FOLDER \  # Path to the folder where the output predictions will be saved
-d DATASET_NAME_OR_ID \  # Dataset ID or name
-c CONFIGURATION \  # Configuration to use (e.g., 2d, 3d_fullres)
--save_probabilities  # Option to save the probabilities

# Example:
nnUNetv2_predict \
-i <nnunet_path>/nnUNet_raw/<dataset_path>/imagesTs/ \
-o <nnunet_path>/nnUNet_results/inference \  
-d 500 -c 2d --save_probabilities  
```

## Run 3D inference | *best model*

```bash
 nnUNetv2_predict \
 -i <path>/7T_NNunet/nnUNet_raw/Dataset500_Segmentation/imagesTs \
 -o <path>/7T_NNunet/nnUNet_results/Dataset500_Segmentation/nnUNetTrainer__nnUNetPlans__3d_fullres/inference \
 -d 500 -c 3d_fullres --save_probabilities
```

## Citation
Cabalo DG, RodrigueZ R, DeKraker J, Kebets V & Bernhardt B.fast_scrubber. Retrieved from osf.io/x95g7

For the segmentation method used 
Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring 
method for deep learning-based biomedical image segmentation. Nature methods, 18(2), 203-211.
