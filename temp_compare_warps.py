import os

import cv2
import matplotlib
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
from vedo import show, Arrows, Spheres, Mesh
from torch.nn import functional as F

from camera import get_sphere_pose
from render import get_smpl_vertices, get_smpl_mesh
from utils import get_rays

camera_transform = get_sphere_pose(0, 0, 2.4)
h, w = 16, 16
f = .5 * w / np.tan(.5 * np.pi / 3)
rays_translation, rays_direction = get_rays(h, w, f, camera_transform)
rays_translation = torch.from_numpy(rays_translation.reshape(-1, 3))
rays_direction = torch.from_numpy(rays_direction.reshape(-1, 3))
warp_by_vertex_mean = False
number_samples = 100
near = 1
far = 4
vertex_sphere_radius = 0.01
std_dev_coarse_sample_prior = 0.03
safe_mode = False

goal_pose = torch.zeros(69).view(1, -1)
goal_pose[0, 38] = np.deg2rad(45)
goal_pose[0, 41] = np.deg2rad(30)

betas = np.array([[-0.3596, -1.0232, -1.7584, -2.0465, 0.3387,
                   -0.8562, 0.8869, 0.5013, 0.5338, -0.0210]])
expression = np.array([[2.7228, -1.8139, 0.6270, -0.5565, 0.3251,
                        0.5643, -1.2158, 1.4149, 0.4050, 0.6516]])

canonical_smpl = torch.from_numpy(get_smpl_vertices(betas, expression))
goal_smpl = torch.from_numpy(get_smpl_vertices(betas=betas, expression=expression, body_pose=goal_pose))
goal_mesh = get_smpl_mesh(body_pose=goal_pose, return_pyrender=False)

t_vals = np.linspace(0., 1., number_samples)
z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))
mids = .5 * (z_vals[1:] + z_vals[:-1])
upper = np.concatenate([mids, z_vals[-1:]], -1)
lower = np.concatenate([z_vals[:1], mids], -1)
# get coarse samples in each bin of the ray
z_vals_simple = torch.from_numpy(lower + (upper - lower) * np.random.rand())

intersector = RayMeshIntersector(goal_mesh)
z_vals_image = []
for ray_index in range(len(rays_translation)):
    intersections = intersector.intersects_location([rays_translation.numpy()[ray_index]],
                                                    [rays_direction.numpy()[ray_index]])
    canonical_intersections_points = torch.from_numpy(intersections[0])  # (N_intersects, 3)

    if len(canonical_intersections_points) == 0:
        z_vals = z_vals_simple
    else:
        mix = D.Categorical(torch.ones(len(canonical_intersections_points), ))
        means = torch.norm(canonical_intersections_points - rays_translation[ray_index], dim=-1)
        comp = D.Normal(means, torch.ones_like(means) * std_dev_coarse_sample_prior)
        gmm = MixtureSameFamily(mix, comp)
        z_vals = gmm.sample((number_samples,))
        z_vals, indices = z_vals.sort()
    z_vals_image.append(z_vals)
z_vals_image = torch.stack(z_vals_image)  # [h*w, number_coarse_samples]
samples = rays_translation[:, None, :] + rays_direction[:, None, :] * z_vals_image[:, :,
                                                                      None]  # [h*w, number_coarse_samples, 3]

warps = []
warps_differentiable = []

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
samples = samples.to(device)
canonical_smpl = canonical_smpl.to(device)
goal_smpl = goal_smpl.to(device)
# iterate through all the samples because we do not have enough memeory to compute all warps at once
print(samples.shape)
for sample_index in tqdm(range(number_samples), desc='Samples'):
    sample = samples[:, sample_index, :]  # [h*w, 3]
    distances = sample[:, None, :].expand((-1, goal_smpl.shape[0], -1)) - goal_smpl[None, :,
                                                                          :]  # [h*w, number_vertices, 3]
    distances = torch.norm(distances, dim=-1)  # [h*w, number_vertices]
    warp = canonical_smpl - goal_smpl  # [number_vertices, 3]


    # attentions = distances  # [h*w, number_vertices]
    # attentions = attentions - vertex_sphere_radius
    # mask = F.relu(-attentions)
    # attentions = torch.exp(attentions)
    # attentions = 1 / attentions
    # warp_differentiable = warp[None, :, :] * attentions[:, :, None] * mask[:, :, None]  # [h*w, number_vertices, 3]
    # warp_differentiable = warp_differentiable.sum(dim=1)  # [h*w, 3]
    # warp_differentiable = warp_differentiable  # [h*w, 3]
    # warps_differentiable.append(warp_differentiable)

    def softmax(x):
        exp = torch.exp(x - torch.max(x))
        return (exp - torch.exp(-torch.max(x))) / exp.sum(-1, keepdim=True)


    # subtract 1/number_vertices after the softmax
    # attentions = distances  # [h*w, number_vertices]
    # attentions = attentions - vertex_sphere_radius
    # attentions = -attentions
    # mask = F.relu(attentions)
    # attentions = mask * attentions
    # attentions = softmax(100000*attentions)
    # attentions = attentions - 1/attentions.shape[-1]
    # print(torch.max(attentions))
    # print(torch.isnan(attentions).any())
    # print(torch.min(attentions))
    # warp_differentiable = warp[None, :, :] * attentions[:, :, None] # [h*w, number_vertices, 3]
    # warp_differentiable = warp_differentiable.sum(dim=1)  # [h*w, 3]
    # warp_differentiable = warp_differentiable  # [h*w, 3]
    # warps_differentiable.append(warp_differentiable)

    attentions = distances  # [h*w, number_vertices]
    attentions = attentions - vertex_sphere_radius
    attentions = -attentions
    attentions = F.relu(attentions)
    attentions = softmax(10000 * attentions)
    warp_differentiable = warp[None, :, :] * attentions[:, :, None]  # [h*w, number_vertices, 3]
    warp_differentiable = warp_differentiable.sum(dim=1)  # [h*w, 3]
    warp_differentiable = warp_differentiable  # [h*w, 3]
    warps_differentiable.append(warp_differentiable)

    if warp_by_vertex_mean:
        assignments = distances  # [h*w, number_vertices]
        outside_sphere = [assignments > vertex_sphere_radius]
        inside_sphere = [assignments < vertex_sphere_radius]

        assignments[outside_sphere] = 0  # [h*w, number_vertices]
        assignments[inside_sphere] = 1  # [h*w, number_vertices]

        warp = warp[None, :, :] * assignments[:, :, None]  # [h*w, number_vertices, 3]
        warp = warp.sum(dim=1)  # [h*w, 3]
        warp = warp / (assignments.sum(dim=1)[:, None] + 1e-10)  # [h*w, 3]
        warps.append(warp)
    else:
        min_indices = torch.argmin(distances, dim=-1)  # [h*w]
        assignments = distances[torch.arange(len(distances)), min_indices]  # [h*w]

        outside_sphere = [assignments > vertex_sphere_radius]
        inside_sphere = [assignments < vertex_sphere_radius]

        assignments[outside_sphere] = 0  # [h*w]
        assignments[inside_sphere] = 1  # [h*w]

        warp = warp[None, :, :].expand(len(assignments), -1, -1)  # [h*w, number_vertices, 3]
        warp = warp[torch.arange(len(warp)), min_indices]  # [h*w, 3]
        warp = warp * assignments[:, None]  # [h*w, 3]
        warps.append(warp)
goal_smpl = goal_smpl.cpu()
canonical_smpl = canonical_smpl.cpu()

warps_differentiable = torch.stack(warps_differentiable, -2).cpu()
warps_differentiable = warps_differentiable.view(-1, 3)
warps = torch.stack(warps, -2).cpu()
warps = warps.view(-1, 3)
samples = samples.view(-1, 3).cpu()
print(samples.shape)
print(warps.shape)

if safe_mode:
    samples = []
    warps = []
    # get bins along the ray
    intersector = RayMeshIntersector(goal_mesh)

    for i, ray_translation in tqdm(enumerate(rays_translation)):
        intersections = intersector.intersects_location([ray_translation.numpy()], [rays_direction[i].numpy()])
        canonical_intersections_points = torch.from_numpy(intersections[0])  # (N_intersects, 3)

        t_vals = np.linspace(0., 1., number_samples)
        z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))
        mids = .5 * (z_vals[1:] + z_vals[:-1])
        upper = np.concatenate([mids, z_vals[-1:]], -1)
        lower = np.concatenate([z_vals[:1], mids], -1)
        # get coarse samples in each bin of the ray
        z_vals = lower + (upper - lower) * np.random.rand()

        ray_samples = ray_translation[None, :] + rays_direction[i][None, :] * z_vals[:, None]
        samples.append(ray_samples)

        distances = ray_samples[:, None, :].expand((-1, goal_smpl.shape[0], -1)) - goal_smpl[None, :,
                                                                                   :]  # [number_samples, number_vertices, 3]
        distances = torch.norm(distances, dim=-1, keepdim=True)

        assignments = distances
        mask_to_0 = [assignments > vertex_sphere_radius]
        mask_to_1 = [assignments < vertex_sphere_radius]
        assignments[mask_to_0] = 0  # [number_samples, number_vertices, 3]
        assignments[mask_to_1] = 1  # [number_samples, number_vertices, 3]

        warp = torch.from_numpy(canonical_smpl.numpy() - goal_smpl.numpy())  # [number_vertices,3]
        warp = warp[None, :, :] * assignments  # [number_samples, number_vertices, 3]
        warp = warp.sum(dim=1)  # [number_samples, number_vertices, 3]
        warp = warp / (assignments.sum(dim=1) + 1e-10)  # [number_samples, 3]

        warps.append(warp)
    warps = torch.cat(warps).view(-1, 3)
    samples = torch.cat(samples).view(-1, 3)

print('show')
smpl_file_name = "SMPLs/smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl"
model = smplx.create(smpl_file_name, model_type='smpl')
output = model(betas=torch.from_numpy(betas).float(), expression=torch.from_numpy(expression).float(),
               return_verts=True)

vertices = output.vertices.detach().cpu().numpy().squeeze()
faces = model.faces
smpl_mesh = Mesh([vertices, faces], alpha=0.3)
images = [[Arrows(samples, samples + warps, s=0.3), Spheres(samples, r=0.004, res=8, alpha=0.1), smpl_mesh],
          [Arrows(samples, samples + warps_differentiable, s=0.3), Spheres(samples, r=0.004, res=8, alpha=0.1),
           smpl_mesh]]
show(images, at=[0, 1])
