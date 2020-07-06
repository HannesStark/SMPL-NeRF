#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from io import BytesIO
import pyrender
import numpy as np
import torch
import smplx
import matplotlib.pyplot as plt
import trimesh
from PIL import Image
import numpy as np
from camera import get_sphere_pose
from utils import get_rays
from trimesh.ray.ray_triangle import RayMeshIntersector
import smplx


def get_smpl_mesh(smpl_file_name: str = None, texture_file_name: str = None,
                  uv_map_file_name: str = None, body_pose: torch.Tensor = None,
                  return_betas_exps=False, return_pyrender=True) -> pyrender.Mesh:
    """
    Load SMPL model, texture file and uv-map.
    Set arm angles and convert to mesh.

    Parameters
    ----------
    smpl_file_name : str
        file name of smpl model (.pkl).
    texture_file_name : str
        file name of texture for smpl (.jpg).
    uv_map_file_name : str
        file name of uv-map for smpl (.npy).
    right_arm_angle : float, optional
        desired right arm angle in radians. The default is 0..
    left_arm_angle : float, optional
        desired left arm angle in radians. The default is 0.
    body_pose : torch.Tensor[1, 69]
        Body poses for SMPL

    Returns
    -------
    mesh : pyrender.Mesh
        SMPL mesh with texture and desired body pose.

    """
    if smpl_file_name is None:
        smpl_file_name = "SMPLs/smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl"
    if texture_file_name is None:
        texture_file_name = "textures/texture.jpg"
    if uv_map_file_name is None:
        uv_map_file_name = "textures/smpl_uv_map.npy"
    model = smplx.create(smpl_file_name, model_type='smpl')
    # set betas and expression to fixed values
    betas = torch.tensor([[-0.3596, -1.0232, -1.7584, -2.0465, 0.3387,
                           -0.8562, 0.8869, 0.5013, 0.5338, -0.0210]])
    expression = torch.tensor([[2.7228, -1.8139, 0.6270, -0.5565, 0.3251,
                                0.5643, -1.2158, 1.4149, 0.4050, 0.6516]])
    
    output = model(betas=betas, expression=expression,
                   return_verts=True, body_pose=body_pose)
    vertices = output.vertices.detach().cpu().numpy().squeeze()
    with open(texture_file_name, 'rb') as file:
        texture = Image.open(BytesIO(file.read()))
    uv = np.load(uv_map_file_name)
    smpl_mesh = trimesh.Trimesh(vertices, model.faces,
                visual=trimesh.visual.TextureVisuals(uv=uv, image=texture),
                                process=False)
    mesh = pyrender.Mesh.from_trimesh(smpl_mesh)
    if not return_pyrender:
        mesh = smpl_mesh
    if return_betas_exps:
        return mesh, betas, expression
    return mesh

def get_smpl_vertices(betas, expression, 
                  smpl_file_name: str = None, texture_file_name: str = None,
                  uv_map_file_name: str = None, body_pose: torch.Tensor = None,
                  return_betas_exps=False) -> np.array:
    """
    Load SMPL model, texture file and uv-map.
    Set arm angles and convert to mesh.

    Parameters
    ----------
    betas: np.array(1, 10)
        Betas for smpl
    expression: np.array(1, 10)
        Expression for smpl
    smpl_file_name : str
        file name of smpl model (.pkl).
    texture_file_name : str
        file name of texture for smpl (.jpg).
    uv_map_file_name : str
        file name of uv-map for smpl (.npy).
    right_arm_angle : float, optional
        desired right arm angle in radians. The default is 0..
    left_arm_angle : float, optional
        desired left arm angle in radians. The default is 0.
    body_pose : torch.Tensor[1, 69]
        Body poses for SMPL

    Returns
    -------
    vertices : np.array
        SMPL mesh with texture and desired body pose.

    """
    if smpl_file_name is None:
        smpl_file_name = "SMPLs/smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl"
    if texture_file_name is None:
        texture_file_name = "textures/texture.jpg"
    if uv_map_file_name is None:
        uv_map_file_name = "textures/smpl_uv_map.npy"
    model = smplx.create(smpl_file_name, model_type='smpl')
    # set betas and expression to fixed values
    betas = torch.tensor(betas).float()
    expression = torch.tensor(expression).float()
    output = model(betas=betas, expression=expression,
                   return_verts=True, body_pose=body_pose)
    vertices = output.vertices.detach().cpu().numpy().squeeze()
    return vertices

def get_human_poses(joints: list, start_angle: list, end_angle: list,
                    number_steps: list) -> list:
    """
    Returns human poses for varying joints in the range [start_angle, end_angle, number_angle]

    Parameters
    ----------
    joints : list
        Joints to vary.
    start_angles : list
        Start angle all joints in degrees.
    end_angles : list
        End angle for all joints in degrees.
    number_steps : list
        Number of angles between start and end for all joints in degrees.

    Returns
    -------
    human_poses : [torch.Tensor[1, 69]] * len(joints)
        Body poses for SMPL with different joint angles

    """
    body_pose = torch.zeros(69).view(1, -1)
    angles = np.linspace(start_angle, end_angle, number_steps)
    human_poses = []
    for i, angle in enumerate(angles):
        body_pose = torch.zeros(69).view(1, -1)
        for joint in joints:
            body_pose[0, joint] = np.deg2rad(angle)
        human_poses.append(body_pose)
    return torch.stack(human_poses)
    
def get_warp(canonical: trimesh.base.Trimesh, goal: trimesh.base.Trimesh,
             camera_transform: np.array, h: int, w: int, 
             camera_angle_x: float, debug: bool=False) -> np.array:
    """
    Calculate warp vectors pointing from goal SMPL to canonical SMPL for the
    closest ray intersection (wrt. camera origin) and return a warp
    for each ray. If the ray doesn't intersect with the goal smpl than
    a warp equal to zero will be returned for that ray (pixel)

    Parameters
    ----------
    canonical : trimesh.base.Trimesh
        Canonical SMPL.
    goal : trimesh.base.Trimesh
        Goal SMPL.
    camera_transform : np.array (4, 4)
        Camera transformation matrix.
    h : int
        Height of camera.
    w : int
        Width of camera.
    camera_angle_x : float
        FOV of camera.
    debug: bool
        If True, a 3D and 2D plot of the image will be created and shown.

    Returns
    -------
    warp_img : np.array (h, w, 3)
        Warp vectors (3D) pointing from goal smpl to canonical smpl
        intersections.

    """
    f = .5 * w / np.tan(.5 * camera_angle_x)
    rays_translation, rays_direction = get_rays(h, w, f, camera_transform)
    camera_origin = rays_translation[0][0]

    # calculate intersections with rays and goal smpl 
    intersector = RayMeshIntersector(goal)
    goal_intersections = intersector.intersects_location(rays_translation.reshape(-1, 3), rays_direction.reshape(-1, 3))
    goal_intersections_points = goal_intersections[0]  # (N_intersects, 3)
    goal_intersections_face_indices = goal_intersections[2]  # (N_intersects, )
    goal_intersections_ray_indices = goal_intersections[1] # (N_intersects, )

    # Find multiple intersections and use only closest
    unique_goal_intersect_points = []
    unique_goal_intersect_face_indices = []
    unique_goal_intersect_ray_indices = []
    intersect_indices = np.arange(len(goal_intersections_points))
    for ray in np.unique(goal_intersections_ray_indices):
        ray_mask = goal_intersections_ray_indices == ray
        indices_ray = intersect_indices[ray_mask]
        ray_intersects = goal_intersections_points[ray_mask]
        distances_camera = np.linalg.norm(ray_intersects-camera_origin, axis=1)
        closest_intersect_index = indices_ray[np.argmin(distances_camera)]
        unique_goal_intersect_points.append(goal_intersections_points[closest_intersect_index])
        unique_goal_intersect_face_indices.append(goal_intersections_face_indices[closest_intersect_index])
        unique_goal_intersect_ray_indices.append(goal_intersections_ray_indices[closest_intersect_index])

    assert(len(unique_goal_intersect_points)==len(unique_goal_intersect_face_indices)==len(unique_goal_intersect_ray_indices))
    assert((np.unique(goal_intersections_ray_indices) == unique_goal_intersect_ray_indices).all())
    goal_intersections_points = np.array(unique_goal_intersect_points)
    goal_intersections_face_indices = np.array(unique_goal_intersect_face_indices)
    goal_intersections_ray_indices = np.array(unique_goal_intersect_ray_indices)
    
    # Calculate for each intersection on goal SMPL the corresponding 
    # intersection on the canonical SMPL
    canonical_intersections = []
    for i, face_idx in enumerate(goal_intersections_face_indices):
        vertex_indices = goal.faces[face_idx]
        goal_vertices = goal.vertices[vertex_indices]
        canonical_vertices = canonical.vertices[vertex_indices]
        lin_coeffs_vertices = np.linalg.solve(goal_vertices.T, goal_intersections_points[i])
        canonical_intersection = canonical_vertices.T.dot(lin_coeffs_vertices)
        canonical_intersections.append(canonical_intersection)
    canonical_intersections = np.array(canonical_intersections)

    # Calculate actual warp for intersections
    warp = canonical_intersections-goal_intersections_points
    # Set each pixel corresponding to ray index to the warp
    warp_img_flat = np.zeros((h*w, 3))
    warp_img_flat[goal_intersections_ray_indices] = warp
    warp_img = warp_img_flat.reshape((h, w, 3))

    warp_min = -1 #np.min(warp_img, axis=(0,1))
    warp_max = 1 #np.max(warp_img, axis=(0,1))
    warp_normalized = (warp_img-warp_min)/(warp_max-warp_min)
    if debug:
        plt.imshow(warp_normalized)
        plt.show()
        scene = pyrender.Scene()
        lines_warp = np.hstack((goal_intersections_points, goal_intersections_points+warp)).reshape(-1, 3)
        primitive = [pyrender.Primitive(lines_warp, mode=1)]
        primitive_mesh = pyrender.Mesh(primitive)
        scene.add(primitive_mesh)
        pyrender.Viewer(scene, use_raymond_lighting=True)
    return warp_img


def render_scene(mesh: pyrender.Mesh, camera_pose: np.array,
                 human_pose: np.array, light_pose: np.array,
                 height: int, width: int, yfov: float, return_depth=False):
    """
    Add mesh, camera and light to scene at desired poses and return rendered
    image.

    Parameters
    ----------
    mesh : pyrender.Mesh
        SMPL mesh.
    camera_pose : np.array (4, 4)
        camera pose matrix in homogeneous coordinates.
    human_pose : np.array (4, 4)
        human pose matrix in homogeneous coordinates..
    light_pose : np.array (4, 4)
        light pose matrix in homogeneous coordinates.
    height : int
        height of image plane of camera in pixels.
    width : int
        width of image plane of camera in pixels.
    yfov : float
        vertical field of view of camera in radians.
    return_depth: bool
        if true the function returns a depth map as well

    Returns
    -------
    img : np.array (height, width, 3)
        rendered image.

    """
    scene = pyrender.Scene()
    scene.add(mesh, pose=human_pose)
    camera = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=1.0)
    scene.add(camera, pose=camera_pose)
    light = pyrender.SpotLight(color=np.ones(3), intensity=200.0,
                                innerConeAngle=np.pi/16.0,
                                outerConeAngle=np.pi/6.0)
    scene.add(light, pose=camera_pose)
    r = pyrender.OffscreenRenderer(height, width)
    img, depth = r.render(scene)
    if return_depth:
        return img, depth
    return img


def save_render(render, f_name):
    """
    Saves render under filename
    """
    plt.figure()
    plt.axis('off')
    plt.imshow(render)
    plt.imsave(f_name, render)
    plt.close()

if __name__ == "__main__":
    pass
