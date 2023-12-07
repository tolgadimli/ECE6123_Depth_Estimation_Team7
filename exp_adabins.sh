#!/bin/bash 
#conda activate dlproject


# Model names:
# UpConv_ResNet50, UpConv_AlexNet, UpConv_VGG16, FCRN_ResNet50, FCRN_AlexNet, FCRN_VGG16

# Dataset names:
# SUN-RGBD, NYUv2

# Loss Function names:
# MSE, berhu

python main.py --model_name 'adabins' --dataset_name 'SUN-RGBD' --loss_fnc 'silog' --lr 3e-4 --wd 1e-2 --bs 4 --device 'cuda:0' 
