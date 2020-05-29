#!/bin/sh

python train.py --dataset_dir=data/camera_0_human_-70_70 --gmm_std=0.005
python train.py --dataset_dir=data/camera_0_human_-70_70 --gmm_std=0.01
python train.py --dataset_dir=data/camera_0_human_-70_70 --gmm_std=0.05
python train.py --dataset_dir=data/camera_0_human_-70_70 --gmm_std=0.1
python train.py --dataset_dir=data/camera_circle_-180_180_human_-70_70
