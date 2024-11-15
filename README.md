# Fast scrubber

## Steps to train model

### 1. Create a conda environment for the nnUNet and follow the installation instructions here: https://github.com/MIC-DKFZ/nnUNet

### 2. Sort your data and paths according to: 
https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/setting_up_paths.md
https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md
https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/set_environment_variables.md

### 4. copy training images & labels, and test images to new directory
	Training images are imagesTr- orig.mgz from the ground truth
	labelsTr are mask- the ground truth images 
	imagesTs- are the testing data set, the origs 

## Run model

### check data
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity >> e.g. nnUNetv2_plan_and_preprocess -d 500 --verify_dataset_integrity

### train data
nnUNetv2_train DATASET_NAME_OR_ID UNET_CONFIGURATION FOLD --npz >> e.g. nnUNetv2_train 500 2d 0 --npz or  nnUNetv2_train 500 3d_fullres 0 --npz –device cuda

>> run it for 5 folds (0-4), and with different configurations (2d, 3d_lowres, 3d_fullres)- 0

### choose best model
nnUNetv2_find_best_configuration DATASET_NAME_OR_ID -c CONFIGURATIONS >> e.g. nnUNetv2_find_best_configuration 500 -c 2d 3d_lowres 3_fullres

### run inference for 2D
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_NAME_OR_ID -c CONFIGURATION --save_probabilities >> nnUNetv2_predict -i <nnunet_path>/nnUNet_raw/<dataset_path>/imagesTs/ -o <nnunet_path>/nnUNet_results/inference -d 500 -c 2d --save_probabilities

## run inference for 3D- this one!
 nnUNetv2_predict -i /host/percy/local_raid/donna/7T_NNunet/nnUNet_raw/Dataset500_Segmentation/imagesTs -o /host/percy/local_raid/donna/7T_NNunet/nnUNet_results/Dataset500_Segmentation/nnUNetTrainer__nnUNetPlans__3d_fullres/inference -d 500 -c 3d_fullres --save_probabilities

