<p align="center">
  <img src="fast_scrubber_logo.png" width="180">
</p>

<h1 align="center">fast_scrubber</h1>

<p align="center">
 Automated surface QC. Goodbye to all-day manual QC!
</p>




# nnUNet for High-Accuracy Brain Mask Prediction

This repository contains the implementation of nnUNet used to train a model for accurately predicting brain masks. The primary motivation for developing this model is to facilitate the removal of meninges from high-resolution T1-weighted images acquired using a 7T MRI protocol (0.5 × 0.5 × 0.5 mm³).

The model was trained on manually quality-controlled masks, carefully reviewed by expert raters to ensure high precision and reliability.

## Steps to train model

### 1. Create a conda environment for the nnUNet and follow the installation instructions here: 
> https://github.com/MIC-DKFZ/nnUNet

### 2. Sort your data and paths according to: 
[setting_up_paths.md](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/setting_up_paths.md)
[dataset_format.md](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md)
[set_environment_variables.md](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/set_environment_variables.md)

### 3. Copy training images & labels, and test images to new directory

| **Category**   | **Description**                               | **Data Type**         |
|----------------|-----------------------------------------------|-----------------------|
| `Training`     | `imagesTr - orig.mgz` from the ground truth   | Training images       |
| `labelsTr`     | `mask-` the ground truth images               | Ground truth labels   |
| `imagesTs`     | Testing dataset, the origs                   | Testing images        |


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
