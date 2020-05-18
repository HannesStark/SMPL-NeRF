import os

import cv2
import smplx
import torch

import torch
import trimesh
from trimesh.ray.ray_triangle import RayMeshIntersector
import numpy as np

a = np.ones(((5,6,3)))
b = np.ones((5))
b = np.repeat(b[np.newaxis,:], 343, axis=0)
print(b)