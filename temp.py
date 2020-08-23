import os
import pickle

import pyrender
import smplx
import trimesh
from torch.utils.data import Subset
import timeit

import numpy as np
import torch
from torchvision.transforms import transforms
from skimage.color import gray2rgb
from trimesh.ray.ray_triangle import RayMeshIntersector

from camera import get_circle_pose, get_pose_matrix, get_sphere_pose
from datasets.rays_from_images_dataset import RaysFromImagesDataset
from datasets.transforms import NormalizeRGB, CoarseSampling, ToTensor
from render import get_smpl_mesh, render_scene, save_render
from utils import get_rays

height, width, yfov = 128, 128, np.pi / 3
camera_radius = 2.4
degrees = np.arange(90, 120, 2)

smpl_file_name = "SMPLs/smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl"
texture_file_name = 'textures/female1.jpg'
uv_map_file_name = 'textures/smpl_uv_map.npy'
save_location = 'baseline/pytorch-CycleGAN-and-pix2pix/datasets/testdir/test'
mesh = get_smpl_mesh(smpl_file_name, texture_file_name, uv_map_file_name)

model = smplx.create(smpl_file_name, model_type='smpl')
# set betas and expression to fixed values
betas = torch.tensor([[-0.3596, -1.0232, -1.7584, -2.0465, 0.3387,
                       -0.8562, 0.8869, 0.5013, 0.5338, -0.0210]])
expression = torch.tensor([[2.7228, -1.8139, 0.6270, -0.5565, 0.3251,
                            0.5643, -1.2158, 1.4149, 0.4050, 0.6516]])
smpl_model = model(betas=betas, expression=expression,
                   return_verts=True)

camera_transform = get_sphere_pose(20, 90, 2.4)
h, w = 128, 128
f = .5 * w / np.tan(.5 * np.pi / 3)
rays_translation, rays_direction = get_rays(h, w, f, camera_transform)
number_samples = 64
near = 2
far = 6

all_rays = []
for i in range(64, 66):
    for j in range(64, 66):
        ray_translation = rays_translation[i][j]
        ray_direction = rays_direction[i][j]
        t_vals = np.linspace(0., 1., number_samples)
        z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))
        mids = .5 * (z_vals[1:] + z_vals[:-1])
        upper = np.concatenate([mids, z_vals[-1:]], -1)
        lower = np.concatenate([z_vals[:1], mids], -1)
        # get coarse samples in each bin of the ray
        z_vals = lower + (upper - lower) * np.random.rand()
        all_rays.append(
            [ray_translation[None, :] + ray_direction[None, :] * z_vals[:, None], ray_translation, ray_direction,
             z_vals, [0.4, 0.3, 0.5]])  # [N_samples, 3]

ray_samples, ray_translation, ray_direction, z_vals, rgb_truth = all_rays[-1]
dependency_rays = all_rays[:-1]
goal_pose = torch.zeros(69).view(1, -1)
goal_pose[0, 38] = 45

betas, expression = smpl_model.betas, smpl_model.expression
goal_model = model(betas=betas, expression=expression,
                   return_verts=True, body_pose=goal_pose)

vertices_canonical = smpl_model.vertices.detach().cpu().numpy().squeeze()
vertices_goal = goal_model.vertices.detach().cpu().numpy().squeeze()

goal_mesh = trimesh.Trimesh(vertices_goal, model.faces, process=False)
canonical_mesh = trimesh.Trimesh(vertices_canonical, model.faces, process=False)

intersector = RayMeshIntersector(canonical_mesh)
intersections = intersector.intersects_location([ray_translation], [ray_direction])
canonical_intersections_points = intersections[0]  # (N_intersects, 3)
intersections_face_indices = intersections[2]  # (N_intersects, )

goal_intersections_points = []
goal_intersections_normals = []
for i, face_idx in enumerate(intersections_face_indices):
    vertex_indices = canonical_mesh.faces[face_idx]
    canonical_vertices = canonical_mesh.vertices[vertex_indices]
    goal_vertices = goal_mesh.vertices[vertex_indices]
    lin_coeffs_vertices = np.linalg.solve(canonical_vertices.T, canonical_intersections_points[i])
    goal_intersection = goal_vertices.T.dot(lin_coeffs_vertices)
    goal_intersections_points.append(goal_intersection)
    goal_intersections_normals.append(goal_mesh.face_normals[face_idx])

goal_intersections_normals = np.array(goal_intersections_normals)
goal_intersections_points = np.array(goal_intersections_points)
canonical_intersections_points = np.array(canonical_intersections_points)


sections_normals = np.split(goal_intersections_normals, len(goal_intersections_normals) / 2)
sections_canonical_points = np.split(canonical_intersections_points, len(canonical_intersections_points) / 2)
sections_goal_points = np.split(goal_intersections_points, len(goal_intersections_points) / 2)

#go through different sections of the ray
for i, section_goal_points in enumerate(sections_goal_points):
    #get virtual ray of section
    virtual_ray_direction = sections_goal_points[i][1] - sections_goal_points[i][0]
    virtual_ray_direction = virtual_ray_direction / np.linalg.norm(virtual_ray_direction)
    virtual_ray_offset = virtual_ray_direction * np.linalg.norm(sections_canonical_points[i][0] - ray_translation)
    virtual_ray_translation = sections_goal_points[i][0] - virtual_ray_offset

    # get camera facing point from our goal intersections (checked by the if statement)
    for j, camera_facing_goal_point in enumerate(section_goal_points):
        if -np.pi / 2 < np.arccos(ray_direction.dot(sections_normals[i][j])) < np.pi / 2:
            camera_facing_canonical_point = sections_canonical_points[i][j]
            print(np.rad2deg(np.arccos(ray_direction.dot(sections_normals[i][j]))))

print('virtual_ray_direction ', virtual_ray_direction)
print('virtual_ray_translation ', virtual_ray_translation)
print('ray_direction ', ray_direction)
print('ray_translation ', ray_translation)

primitive2 = pyrender.Primitive([virtual_ray_translation, virtual_ray_translation + virtual_ray_direction * 6], mode=3,
                                color_0=[0., 0.8, 0.8, 1])
primitive = pyrender.Primitive([ray_translation, ray_translation + ray_direction * 6], mode=3)
prim_normal1 = pyrender.Primitive(
    [goal_intersections_points[0], goal_intersections_points[0] + goal_intersections_normals[0]], mode=3)
prim_normal2 = pyrender.Primitive(
    [goal_intersections_points[1], goal_intersections_points[1] + goal_intersections_normals[1]], mode=3)
lines = pyrender.Mesh([primitive, primitive2, prim_normal1, prim_normal2])
mesh = pyrender.Mesh.from_trimesh(canonical_mesh)
scene = pyrender.Scene()
scene.add(mesh)
scene.add(lines)
pyrender.Viewer(scene, use_raymond_lighting=True)

mesh = pyrender.Mesh.from_trimesh(goal_mesh)
scene = pyrender.Scene()
scene.add(mesh)
scene.add(lines)
pyrender.Viewer(scene, use_raymond_lighting=True)
