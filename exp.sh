#!/bin/bash 
#conda activate dlproject


# Model names:
# UpConv_ResNet50, UpConv_AlexNet, UpConv_VGG16, FCRN_ResNet50, FCRN_AlexNet, FCRN_VGG16

# Dataset names:
# SUN-RGBD, NYUv2

# Loss Function names:
# MSE, berhu

# python main.py --model_name 'UpConv_VGG16' --dataset_name 'NYUv2' --loss_fnc 'MSE' --device 'cuda:0' & \
# python main.py --model_name 'UpConv_VGG16' --dataset_name 'SUN-RGBD' --loss_fnc 'MSE' --device 'cuda:1' & \
# python main.py --model_name 'UpConv_VGG16' --dataset_name 'NYUv2' --loss_fnc 'berhu' --device 'cuda:2' & \
# python main.py --model_name 'UpConv_VGG16' --dataset_name 'SUN-RGBD' --loss_fnc 'berhu' --device 'cuda:3'


# python main.py --model_name 'UpConv_ResNet50' --dataset_name 'NYUv2' --loss_fnc 'MSE' --device 'cuda:0' & \
# python main.py --model_name 'UpConv_ResNet50' --dataset_name 'NYUv2' --loss_fnc 'berhu' --device 'cuda:1' & \
# python main.py --model_name 'FCRN_ResNet50' --dataset_name 'NYUv2' --loss_fnc 'MSE' --device 'cuda:2' & \
# python main.py --model_name 'FCRN_ResNet50' --dataset_name 'NYUv2' --loss_fnc 'berhu' --device 'cuda:3'
# python main.py --model_name 'FCRN_ResNet50' --dataset_name 'NYUv2' --loss_fnc 'berhu' --device 'cuda:2' & \
# python main.py --model_name 'FCRN_ResNet50' --dataset_name 'SUN-RGBD' --loss_fnc 'berhu' --device 'cuda:3'


# python main.py --model_name 'FCRN_ResNet50v2' --decoder 'upproj' --dataset_name 'NYUv2' --loss_fnc 'l1' --device 'cuda:0' & \
# python main.py --model_name 'FCRN_ResNet50v2' --decoder 'upproj' --dataset_name 'NYUv2' --loss_fnc 'berhu' --device 'cuda:1' & \
# python main.py --model_name 'FCRN_ResNet50v2' --decoder 'fasterupproj' --dataset_name 'NYUv2' --loss_fnc 'l1' --device 'cuda:2' & \
# python main.py --model_name 'FCRN_ResNet50v2' --decoder 'fasterupproj' --dataset_name 'NYUv2' --loss_fnc 'berhu' --device 'cuda:3'


python main.py --model_name 'FCRN_ResNet50v2' --decoder 'upproj' --dataset_name 'SUN-RGBD' --loss_fnc 'l1' --device 'cuda:0' & \
python main.py --model_name 'FCRN_ResNet50v2' --decoder 'upproj' --dataset_name 'SUN-RGBD' --loss_fnc 'berhu' --device 'cuda:1' & \
python main.py --model_name 'FCRN_ResNet50v2' --decoder 'fasterupproj' --dataset_name 'SUN-RGBD' --loss_fnc 'l1' --device 'cuda:2' & \
python main.py --model_name 'FCRN_ResNet50v2' --decoder 'fasterupproj' --dataset_name 'SUN-RGBD' --loss_fnc 'berhu' --device 'cuda:3'