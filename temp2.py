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


def get_dependent_rays(ray_translation: np.array, ray_direction: np.array,
                       canonical: trimesh.base.Trimesh, goal: trimesh.base.Trimesh,
                       camera_transform: np.array, h: int, w: int, f: float) -> np.array:
    """
    Takes one ray (with translation + direction) and returns all dependent 
    rays (as camera pixels) and an empty list if there is no dependent ray.


    Parameters
    ----------
    ray_translation : np.array
        Point on orgin of ray.
    ray_direction : np.array
        Direction of ray.
    canonical : trimesh.base.Trimesh
        Trimesh of SMPL in canonical pose.
    goal : trimesh.base.Trimesh
        Trimesh of SMPL in goal pose.
    camera_transform : np.array
        World to Camera transformation.
    h : int
        Height of image.
    w : int
        Width of image.
    f : float
        Focal length of camera.

    Returns
    -------
    list(np.array)
        Camera pixels of dependent rays.

    """
    
    intersector = RayMeshIntersector(canonical)
    intersections = intersector.intersects_location([ray_translation], [ray_direction])
    intersections_points = intersections[0] # (N_intersects, 3)
    intersections_face_indices = intersections[2] # (N_intersects, )
    if len(intersections_face_indices) == 0:
        return []  # Return  empty list if there are no dependent rays

    
    goal_intersections = []
    vertices = []
    for i, face_idx in enumerate(intersections_face_indices):
        vertex_indices = canonical.faces[face_idx]
        canonical_vertices = canonical.vertices[vertex_indices]
        goal_vertices = goal.vertices[vertex_indices]
        lin_coeffs_vertices = np.linalg.solve(canonical_vertices, intersections_points[i])
        goal_intersection = goal_vertices.dot(lin_coeffs_vertices)
        goal_intersections.append(goal_intersection)
        vertices.append(vertex_indices) # For painting human
    goal_intersections = np.array(goal_intersections)
    rot_1 = R.from_euler('xyz', [0, 180, 0], degrees=True).as_matrix()
    rot_2 = R.from_euler('xyz', [0, 0, 180], degrees=True).as_matrix()
    goal_intersections = goal_intersections - camera_transform[:3, 3] # This translates the intersections  --> Now the intersections are in the camera frame
    world2camera = rot_2.dot(rot_1.dot(camera_transform[:3, :3].T)) # rot_2 after rot_1 after camera_transform
    goal_intersections = np.dot(world2camera, goal_intersections.T).T # This rotates the intersections with the camera rotation matrix
    
    rvec , tvec = np.zeros(3), np.zeros(3) # Now no further trafo is needed
    camera_matrix = np.array([[f, 0.0, w / 2],
                              [0.0, f, h / 2],
                              [0.0, 0.0, 1.0]])
    distortion_coeffs = np.array([0.0, 0.0, 0.0, 0.0])
    camera_coords = cv2.projectPoints(goal_intersections, rvec, tvec, camera_matrix, distortion_coeffs)[0]
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
camera_transform = get_sphere_pose(0, 90, 1)
img_coord_1 = 64

rays_translation, rays_direction = get_rays(h, w, f, camera_transform)

image_coordinates1, vertices1 = get_dependent_rays(rays_translation[img_coord_1][img_coord_1], rays_direction[img_coord_1][img_coord_1], mesh_canonical,
                                                   mesh_canonical, camera_transform, h, w, f)
image_coordinates2, vertices2 = get_dependent_rays(rays_translation[64][67], rays_direction[64][67], mesh_canonical,
                                                   mesh_canonical, camera_transform, h, w, f)
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
