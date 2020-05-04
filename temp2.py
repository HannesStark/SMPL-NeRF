import pickle

import cv2
import numpy as np
import pyrender
import smplx
import torch
import trimesh

from camera import get_pose_matrix, get_circle_pose
from inference import inference
from trimesh.ray.ray_triangle import RayMeshIntersector

from utils import get_rays


def get_dependent_rays(ray_translation, ray_direction, canonical, goal, camera_transform, h, w, f):
    intersector = RayMeshIntersector(canonical)
    intersection_faces = intersector.intersects_location([ray_translation], [ray_direction])[2]

    if len(intersection_faces) == 0:
        return []  # Return  empty list if there are no dependent rays

    rvec = cv2.Rodrigues(camera_transform[:3, :3])
    rvec = np.array(rvec[0], dtype=float)
    tvec = np.array(camera_transform[:3, 3], dtype=float)
    print(rvec)
    print(tvec)
    camera_matrix = np.array([[f, 0.0, w / 2],
                              [0.0, f, h / 2],
                              [0.0, 0.0, 1.0]])
    distortion_coeffs = np.array([0.0, 0.0, 0.0, 0.0])
    goal_intersections = []
    vertices = []
    print('intersection faces', intersection_faces)
    for face_idx in intersection_faces:
        vertex_indices = canonical.faces[face_idx]
        goal_vertices = goal.vertices[vertex_indices]
        vertices.append(vertex_indices)
        goal_intersection = np.sum(np.array(goal_vertices), axis=0) / 3
        goal_intersections.append(goal_intersection)
    print(goal_intersections)

    camera_coords = cv2.projectPoints(np.array(goal_intersections), rvec, tvec, camera_matrix, distortion_coeffs)[0]
    return np.round(camera_coords.reshape(-1, 2)), vertices


smpl_file_name = "SMPLs/smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl"

model = smplx.create(smpl_file_name, model_type='smpl')
# set betas and expression to fixed values
betas = torch.tensor([[-0.3596, -1.0232, -1.7584, -2.0465, 0.3387,
                       -0.8562, 0.8869, 0.5013, 0.5338, -0.0210]])
expression = torch.tensor([[2.7228, -1.8139, 0.6270, -0.5565, 0.3251,
                            0.5643, -1.2158, 1.4149, 0.4050, 0.6516]])
body_pose = torch.zeros(69).view(1, -1)
body_pose[0, 41] = np.deg2rad(45)
body_pose[0, 38] = np.deg2rad(45)
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
camera_transform = get_circle_pose(90, 2.4)
rays_translation, rays_direction = get_rays(h, w, f, camera_transform)
print(rays_translation.shape)

image_coordinates1, vertices1 = get_dependent_rays(rays_translation[64][64], rays_direction[64][64], mesh_canonical, mesh_canonical, camera_transform, h, w, f)
image_coordinates2, vertices2 = get_dependent_rays(rays_translation[64][67], rays_direction[64][67], mesh_canonical, mesh_canonical, camera_transform, h, w, f)
print(image_coordinates1)
print(image_coordinates2)

print(mesh_canonical.vertices.shape)
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