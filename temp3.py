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
from vedo import show, Arrows, Spheres

from camera import get_sphere_pose
from render import get_smpl_vertices, get_smpl_mesh
from utils import get_rays

camera_transform = get_sphere_pose(0, 0, 2.4)
h, w = 8, 8
f = .5 * w / np.tan(.5 * np.pi / 3)
rays_translations, rays_directions = get_rays(h, w, f, camera_transform)
rays_translations = torch.from_numpy(rays_translations.reshape(-1, 3))
rays_directions = torch.from_numpy(rays_directions.reshape(-1, 3))
number_samples = 64
near = 1
far = 4
vertex_radius = 0.07

goal_pose = torch.zeros(69).view(1, -1)
goal_pose[0, 38] = 45

betas = np.array([[-0.3596, -1.0232, -1.7584, -2.0465, 0.3387,
                       -0.8562, 0.8869, 0.5013, 0.5338, -0.0210]])
expression = np.array([[2.7228, -1.8139, 0.6270, -0.5565, 0.3251,
                            0.5643, -1.2158, 1.4149, 0.4050, 0.6516]])

canonical_smpl = get_smpl_vertices(betas, expression)
goal_smpl = get_smpl_vertices(betas=betas, expression=expression, body_pose=goal_pose)
goal_mesh = get_smpl_mesh(body_pose=goal_pose, return_pyrender=False)

all_samples = []
all_warped_samples = []
# get bins along the ray
intersector = RayMeshIntersector(goal_mesh)

samples = []
warps = []
for i, ray_translation in tqdm(enumerate(rays_translations)):
    intersections = intersector.intersects_location([ray_translation.numpy()], [rays_directions[i].numpy()])
    canonical_intersections_points = torch.from_numpy(intersections[0])  # (N_intersects, 3)

    #if len(canonical_intersections_points) == 0:
    #    t_vals = np.linspace(0., 1., number_samples)
    #    z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))
    #    mids = .5 * (z_vals[1:] + z_vals[:-1])
    #    upper = np.concatenate([mids, z_vals[-1:]], -1)
    #    lower = np.concatenate([z_vals[:1], mids], -1)
    #    # get coarse samples in each bin of the ray
    #    z_vals = lower + (upper - lower) * np.random.rand()
    #else:
    #    mix = D.Categorical(torch.ones(len(canonical_intersections_points), ))
    #    means = torch.norm(canonical_intersections_points - ray_translation, dim=-1)
    #    comp = D.Normal(means, torch.ones_like(means) * 0.07)
    #    gmm = MixtureSameFamily(mix, comp)
    #    z_vals = gmm.sample((number_samples,))

    t_vals = np.linspace(0., 1., number_samples)
    z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))
    mids = .5 * (z_vals[1:] + z_vals[:-1])
    upper = np.concatenate([mids, z_vals[-1:]], -1)
    lower = np.concatenate([z_vals[:1], mids], -1)
    # get coarse samples in each bin of the ray
    z_vals = lower + (upper - lower) * np.random.rand()

    ray_samples = ray_translation[None, :] + rays_directions[i][None, :] * z_vals[:, None]
    samples.append(ray_samples)

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

    warps.append(warp)

warps = torch.cat(warps).view(-1, 3)
samples = torch.cat(samples).view(-1, 3)

print('show')
show([Arrows(samples, samples +warps, s=0.3), Spheres(samples, r=0.01, res=8)], at=0)



