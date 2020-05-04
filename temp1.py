import cv2
import smplx
import torch

import torch
import trimesh
from trimesh.ray.ray_triangle import RayMeshIntersector
import numpy as np

means = 0
num_images = 15
fakes = []
reals = []
for i in range(num_images):
    fake = cv2.imread(
        'baseline/pytorch-CycleGAN-and-pix2pix/results/SMPL_pix2pix/test_latest/images/img_{:03d}_fake_B.png'.format(i))
    real = cv2.imread(
        'baseline/pytorch-CycleGAN-and-pix2pix/results/SMPL_pix2pix/test_latest/images/img_{:03d}_real_B.png'.format(i))
    fake = np.array(fake).astype(int)
    real = np.array(real).astype(int)
    fakes.append(fake)
    reals.append(real)
fakes = np.array(fakes)
reals = np.array(reals)
renders = fakes.reshape((len(fakes), -1))
rerenders = reals.reshape((len(reals), -1))
print(np.mean(np.abs(renders - rerenders)))
