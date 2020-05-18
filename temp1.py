import os

import cv2
import smplx
import torch

import torch
import trimesh
from trimesh.ray.ray_triangle import RayMeshIntersector
import numpy as np

from datasets.smpl_nerf_dataset import SmplNerfDataset
from datasets.transforms import CoarseSampling

transform = transforms.Compose(
        [NormalizeRGB(), CoarseSampling(2, 6, 64), ToTensor()])

train_dir = 'data'
dataset = SmplNerfDataset(train_dir, os.path.join(train_dir, 'transforms.pkl'), transform)
