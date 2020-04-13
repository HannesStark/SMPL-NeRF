#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from io import BytesIO
import pyrender
import numpy as np
import os
import torch
import smplx
import matplotlib.pyplot as plt
import trimesh
import pickle
from PIL import Image

from camera import get_pose_matrix, get_sphere_pose
from utils import disjoint_indices

np.random.seed(0)

def get_smpl_mesh():
    """Load SMPL model and convert it to mesh
    """
    smpl_path = "SMPLs/smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl"
    model = smplx.create(smpl_path, model_type='smpl')
    betas = torch.tensor([[-0.3596, -1.0232, -1.7584, -2.0465, 0.3387, -0.8562, 0.8869, 0.5013,
                           0.5338, -0.0210]], dtype=torch.float32)
    expression = torch.tensor([[2.7228, -1.8139, 0.6270, -0.5565, 0.3251, 0.5643, -1.2158, 1.4149,
                                0.4050, 0.6516]])
    body_pose = torch.zeros(69).view(1, -1)
    body_pose[0, 41] = 0 # right arm angle 
    body_pose[0, 38] = 0 # left arm angle 
    output = model(betas=betas, expression=expression,
                   return_verts=True, body_pose=body_pose)
    vertices = output.vertices.detach().cpu().numpy().squeeze()

    with open('textures/texture.jpg', 'rb') as file:
        texture = Image.open(BytesIO(file.read()))

    uv = np.load('textures/smpl_uv_map.npy')
    smpl_mesh = trimesh.Trimesh(vertices, model.faces, 
                visual=trimesh.visual.TextureVisuals(uv=uv, image=texture),
                                process=False)

    mesh = pyrender.Mesh.from_trimesh(smpl_mesh)
    return mesh


def render_scene(camera_pose: np.array, human_pose: np.array, 
                 light_pose: np.array, height: int, width: int, yfov: float):
    """Render Scene and return image
    
        Args:
            camera_pose (np.array): camera pose matrix
            human_pose (np.array): human pose matrix
            light_pose (np.array): light pose matrix
            height (int): height of rendered image
            width (int): width of rendered image
            yfov (float): vertical field of view in radians
            
    """
    mesh = get_smpl_mesh()
    scene = pyrender.Scene()
    scene.add(mesh, pose=human_pose)
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3, aspectRatio=1.0)
    print(camera_pose)
    scene.add(camera, pose=camera_pose)
    light = pyrender.SpotLight(color=np.ones(3), intensity=200.0,
                                innerConeAngle=np.pi/16.0,
                                outerConeAngle=np.pi/6.0)
    scene.add(light, pose=camera_pose)
    r = pyrender.OffscreenRenderer(height, width)
    color, depth = r.render(scene)
    return color


def save_render(render, f_name):
    """Save rendered image
      
        Args: 
            render(np.array): rendered image
            f_name(string): file name
    """
    
    plt.figure()
    plt.axis('off')
    plt.imshow(render)
    plt.imsave(f_name, render)

def render(camera_phi, camera_theta, camera_radius=2.4, 
         human_phi=0, human_theta=0, 
         height=512, width=512, yfov=np.pi / 3):
    """ Compute human and camera pose, render scene and save render
        
        Args:
            camera_phi (float): rotation of camera around x axis in degrees
            camera_theta (float): rotation of camera around y axis in degrees
            camera_radius (float): camera displacement from origin
            human_phi (float): rotation of human around x axis in degrees
            human_theta (float): rotation of human around y axis in degrees
            height (int): height of rendered image
            width (int): width of rendered image
            yfov (float): vertical field of view in radians
    """
    human_pose = get_pose_matrix(phi=human_phi, theta=human_theta)
    camera_pose = get_sphere_pose(camera_phi, camera_theta, camera_radius)
    rgb = render_scene(camera_pose, human_pose, camera_pose, height, width,
                          yfov)
    return rgb, camera_pose
    
def render_sphere(start_angle, end_angle, number_steps, height, width, 
                  camera_radius, yfov, save_dir, train_val_ratio=0.8):
    camera_phis = np.linspace(start_angle, end_angle, number_steps)
    camera_thetas = np.linspace(start_angle, end_angle, number_steps)
    index = 0
    train_image_transform_map = {}
    val_image_transform_map = {}
    size = number_steps ** 2
    train_indices, val_indices = disjoint_indices(size, train_val_ratio)
    train_dir, val_dir = os.path.join(save_dir, "train"), os.path.join(save_dir, "val")
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
        os.mkdir(val_dir)
    for camera_phi in camera_phis:
        for camera_theta in camera_thetas:
            image_name = "img_{:03d}.png".format(index)
            rgb, camera_pose = render(camera_phi, camera_theta, camera_radius=camera_radius,
                   height=height, width=width)
            if index in train_indices:
                img_file_name = os.path.join(train_dir, image_name)
                train_image_transform_map[image_name] = camera_pose
            else:
                img_file_name = os.path.join(val_dir, image_name)
                val_image_transform_map[image_name] = camera_pose
            
            save_render(rgb, img_file_name) 
            index += 1
    camera_angle_x = yfov / 2 # camera angle x is half field of view 
    train_dict = {'camera_angle_x': camera_angle_x, 'image_transform_map': train_image_transform_map}
    val_dict = {'camera_angle_x': camera_angle_x, 'image_transform_map': val_image_transform_map}
    train_pkl_file_name = os.path.join(train_dir, 'transforms.pkl')
    with open(train_pkl_file_name, 'wb') as handle:
        pickle.dump(train_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    val_pkl_file_name = os.path.join(val_dir, 'transforms.pkl')
    with open(val_pkl_file_name, 'wb') as handle:
        pickle.dump(val_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
   
if __name__ == "__main__":
    save_dir = "data"
    render_sphere(start_angle=-20, end_angle=20, number_steps=10, height=512, width=512, 
                  camera_radius=2.4, yfov=np.pi / 3, save_dir=save_dir)
    with open(os.path.join(save_dir, "train", 'transforms.pkl'), 'rb') as handle:
        transforms_dict = pickle.load(handle)
    