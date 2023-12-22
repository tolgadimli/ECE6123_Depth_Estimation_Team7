#!/bin/bash 

python codes/main.py --model_name 'DORN' --loss_fnc 'ordinal' --bs 3 --wd 5e-4 --lr 1e-4 --dataset_name 'NYUv2' --device 'cuda:1' & \
python codes/main.py --model_name 'DORN' --loss_fnc 'ordinal' --bs 3 --wd 5e-4 --lr 1e-4 --dataset_name 'SUN-RGBD' --device 'cuda:2' & \
python codes/main.py --model_name 'DORN' --loss_fnc 'ordinal' --bs 3 --wd 5e-4 --lr 1e-4 --dataset_name 'DIODE' --device 'cuda:3' 

