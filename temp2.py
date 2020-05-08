import pickle

import cv2
import numpy as np
import pyrender
import smplx
import torch
import trimesh

from scipy.spatial.transform import Rotation as R
from camera import get_pose_matrix, get_circle_pose, get_sphere_pose, get_xyzphitheta
from inference import inference
from trimesh.ray.ray_triangle import RayMeshIntersector

from utils import get_rays





smpl_file_name = "SMPLs/smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl"

model = smplx.create(smpl_file_name, model_type='smpl')
# set betas and expression to fixed values
betas = torch.tensor([[-0.3596, -1.0232, -1.7584, -2.0465, 0.3387,
                       -0.8562, 0.8869, 0.5013, 0.5338, -0.0210]])
expression = torch.tensor([[2.7228, -1.8139, 0.6270, -0.5565, 0.3251,
                            0.5643, -1.2158, 1.4149, 0.4050, 0.6516]])
body_pose = torch.zeros(69).view(1, -1)
body_pose[0, 41] = np.deg2rad(45)
body_pose[0, 38] = np.deg2rad(-45)
canonical = model(betas=betas, expression=expression,
                  return_verts=True)
goal = model(betas=betas, expression=expression,
             return_verts=True, body_pose=body_pose)
vertices_canonical = canonical.vertices.detach().cpu().numpy().squeeze()
vertices_goal = goal.vertices.detach().cpu().numpy().squeeze()

mesh_canonical = trimesh.Trimesh(vertices_canonical, model.faces, process=False)
mesh_goal = trimesh.Trimesh(vertices_goal, model.faces, process=False)

origin = [[2.4, 0, 0]]
direction = [[-1, 0, 0.0]]

h, w = 128, 128
f = .5 * w / np.tan(.5 * np.pi / 3)
camera_transform = get_sphere_pose(20, 90, 2.4)
img_coord_1 = 64

rays_translation, rays_direction = get_rays(h, w, f, camera_transform)

image_coordinates1, vertices1 = get_dependent_rays_indices(rays_translation[img_coord_1][img_coord_1], rays_direction[img_coord_1][img_coord_1], mesh_canonical,
                                                           mesh_goal, camera_transform, h, w, f)
image_coordinates2, vertices2 = get_dependent_rays_indices(rays_translation[64][67], rays_direction[64][67], mesh_canonical,
                                                           mesh_goal, camera_transform, h, w, f)
print("Original Image coords for starting ray: ", img_coord_1)
print("Reprojected image_coordinates1: ", image_coordinates1)
print("Reprojected image_coordinates2: ", image_coordinates2)

vertex_colors = np.ones([mesh_canonical.vertices.shape[0], 4]) * [0.7, 0.7, 0.7, 1]
for vertex in np.concatenate(vertices1):
    vertex_colors[vertex] = [0, 0, 0, 1]

for vertex in np.concatenate(vertices2):
    vertex_colors[vertex] = [0, 1, 1, 1]
tri_mesh = trimesh.Trimesh(mesh_canonical.vertices, model.faces,
                           vertex_colors=vertex_colors)
mesh = pyrender.Mesh.from_trimesh(tri_mesh)
scene = pyrender.Scene()
scene.add(mesh)
pyrender.Viewer(scene, use_raymond_lighting=True)
tri_mesh = trimesh.Trimesh(mesh_goal.vertices, model.faces,
                           vertex_colors=vertex_colors)
mesh = pyrender.Mesh.from_trimesh(tri_mesh)
scene = pyrender.Scene()
scene.add(mesh)
pyrender.Viewer(scene, use_raymond_lighting=True)
