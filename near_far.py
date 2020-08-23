import os

import cv2
import pyrender
import smplx
import torch

import torch
import trimesh
import trimesh as trimesh
from torch.distributions import MixtureSameFamily
import torch.distributions as D
from tqdm import tqdm
from trimesh.ray.ray_triangle import RayMeshIntersector
import numpy as np

from camera import get_sphere_pose, get_circle_on_sphere_poses
from render import get_smpl_vertices, get_smpl_mesh
from util.smpl_sequence_loading import load_pose_sequence
from utils import get_rays

h, w = 8, 8
f = .5 * w / np.tan(.5 * np.pi / 3)
number_samples = 64
vertex_radius = 0.07
betas = np.array([[-0.3596, -1.0232, -1.7584, -2.0465, 0.3387,
                       -0.8562, 0.8869, 0.5013, 0.5338, -0.0210]])
expression = np.array([[2.7228, -1.8139, 0.6270, -0.5565, 0.3251,
                            0.5643, -1.2158, 1.4149, 0.4050, 0.6516]])
camera_radius = 2.4
number_steps = 20

all_samples = []



poses, _ = load_pose_sequence('data/Walk B10 - Walk turn left 45_poses.npz', device='cpu')
near = 10
far = 0
for pose in  tqdm(poses):
    camera_transforms, _ = get_circle_on_sphere_poses(number_steps, 20,camera_radius)
    vertices = get_smpl_vertices(betas=betas, expression=expression, body_pose=pose)
    for camera_transform in camera_transforms:
        rays_translations, ray_directions = get_rays(h, w, f, camera_transform)
        camera_location = rays_translations[0][0]
        distances = ((vertices - camera_location) @ (-camera_location))/np.linalg.norm(camera_location)
        max = np.max(distances)
        min = np.min(distances)
        if max > far:
            far = max
        if min < near:
            near = min

print('near: ', near)
print('far: ', far)





