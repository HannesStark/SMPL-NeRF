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


def get_smpl_mesh(smpl_file_name: str, texture_file_name: str,
                  uv_map_file_name: str, right_arm_angle: float = 0.,
                  left_arm_angle: float = 0) -> pyrender.Mesh:
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

    Returns
    -------
    mesh : pyrender.Mesh
        SMPL mesh with texture and desired body pose.

    """
    model = smplx.create(smpl_file_name, model_type='smpl')
    # set betas and expression to fixed values
    betas = torch.tensor([[-0.3596, -1.0232, -1.7584, -2.0465, 0.3387,
                           -0.8562, 0.8869, 0.5013, 0.5338, -0.0210]])
    expression = torch.tensor([[2.7228, -1.8139, 0.6270, -0.5565, 0.3251,
                                0.5643, -1.2158, 1.4149, 0.4050, 0.6516]])
    body_pose = torch.zeros(69).view(1, -1)
    body_pose[0, 41] = right_arm_angle
    body_pose[0, 38] = left_arm_angle
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
    return mesh


def render_scene(mesh: pyrender.Mesh, camera_pose: np.array,
                 human_pose: np.array, light_pose: np.array,
                 height: int, width: int, yfov: float):
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
