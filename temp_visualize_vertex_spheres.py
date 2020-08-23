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

from camera import get_sphere_pose
from render import get_smpl_vertices, get_smpl_mesh
from util.smpl_sequence_loading import load_pose_sequence
from utils import get_rays

camera_transform = get_sphere_pose(0, 0, 2.4)
print(camera_transform)
h, w = 8, 8
f = .5 * w / np.tan(.5 * np.pi / 3)
rays_translations, ray_directions = get_rays(h, w, f, camera_transform)
rays_translations = torch.from_numpy(rays_translations.reshape(-1, 3))
ray_directions = torch.from_numpy(ray_directions.reshape(-1, 3))
number_samples = 64
near = 1.5
far = 3
vertex_radius = 0.07

goal_pose = torch.zeros(69).view(1, -1)
goal_pose[0, 38] = 45

betas = torch.tensor([[-0.3596, -1.0232, -1.7584, -2.0465, 0.3387,
                       -0.8562, 0.8869, 0.5013, 0.5338, -0.0210]])
expression = torch.tensor([[2.7228, -1.8139, 0.6270, -0.5565, 0.3251,
                            0.5643, -1.2158, 1.4149, 0.4050, 0.6516]])

canonical_smpl = get_smpl_vertices(betas, expression)
goal_smpl = get_smpl_vertices(betas=betas, expression=expression, body_pose=goal_pose)
goal_mesh = get_smpl_mesh(body_pose=goal_pose, return_pyrender=False)

all_samples = []
all_warped_samples = []
# get bins along the ray


for i, ray_translation in tqdm(enumerate(rays_translations)):


    t_vals = np.linspace(0., 1., number_samples)
    z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))
    mids = .5 * (z_vals[1:] + z_vals[:-1])
    upper = np.concatenate([mids, z_vals[-1:]], -1)
    lower = np.concatenate([z_vals[:1], mids], -1)
    # get coarse samples in each bin of the ray
    z_vals = lower + (upper - lower) * np.random.rand()

    ray_samples = ray_translation[None, :] + ray_directions[i][None, :] * z_vals[:, None]
    all_samples.append(ray_samples)

    distances = ray_samples[:, None, :].expand((-1, goal_smpl.shape[0], -1)) - goal_smpl[None, :,
                                                                               :]  # [number_samples, number_vertices, 3]
    distances = torch.norm(distances, dim=-1, keepdim=True)

    assignments = distances
    mask_to_0 = [assignments > vertex_radius]
    mask_to_1 = [assignments < vertex_radius]
    assignments[mask_to_0] = 0  # [number_samples, number_vertices, 3]
    assignments[mask_to_1] = 1  # [number_samples, number_vertices, 3]

    warp = torch.from_numpy(canonical_smpl - goal_smpl)  # [number_vertices,3]
    warp = warp[None, :, :] * assignments  # [number_samples, number_vertices, 3]
    warp = warp.sum(dim=1)  # [number_samples, number_vertices, 3]
    warp = warp / (assignments.sum(dim=1) + 1e-10)  # [number_samples, 3]

    warped_samples = ray_samples + warp

    all_warped_samples.append(warped_samples)
all_warped_samples = torch.cat(all_warped_samples).view(-1, 3)
all_samples = torch.cat(all_samples).view(-1, 3)





poses, _ = load_pose_sequence('data/G14-  roundhouse body left_poses.npz', device='cpu')


sm1 = trimesh.creation.uv_sphere(radius=0.007)
sm1.visual.vertex_colors = [1.0, 1.0, 0.0]
tfs1 = np.tile(np.eye(4), (len(all_samples), 1, 1))
tfs1[:, :3, 3] = all_samples
m1 = pyrender.Mesh.from_trimesh(sm1, poses=tfs1)

for pose in  poses:
    vertices = get_smpl_vertices(betas=betas, expression=expression, body_pose=pose)
    sm = trimesh.creation.uv_sphere(radius=0.01)
    sm.visual.vertex_colors = [1.0, 0.0, 0.0]
    tfs = np.tile(np.eye(4), (len(vertices), 1, 1))
    tfs[:, :3, 3] = vertices
    m = pyrender.Mesh.from_trimesh(sm, poses=tfs)

    scene = pyrender.Scene()
    scene.add(m)
    scene.add(m1)
    pyrender.Viewer(scene, use_raymond_lighting=True)




