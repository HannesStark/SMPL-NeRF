# -*- coding: utf-8 -*-
import math
from typing import Tuple

import pyrender
import numpy as np
import os.path as osp
import argparse

import torch

import smplx
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import pyrender
import trimesh


def get_pose_matrix(x=0, y=0, z=0, phi=0, theta=0, psi=0):
    rot = R.from_euler('xyz', [phi, theta, psi], degrees=True).as_matrix()
    trans = np.array([[x, y, z]])
    pose = np.concatenate((np.concatenate((rot, trans.T), axis=1),
                           [[0, 0, 0, 1]]), axis=0)
    return pose


def camera_origin_direction(x: float, y: float, z: float) -> Tuple[float, float]:
    """Calculates phi and theta in degrees for a camera too face the origin of the coordinate system

        Args:
            x (float): x coordinate of camera
            y (float): y coordinate of camera
            z (float): z coordinate of camera
        """
    phi = np.degrees(np.arctan2(y, z))
    theta = np.degrees(np.arctan2(x, z))
    return phi.item(), theta.item()


model = smplx.create("models", model_type='smplx')
camera_phi, camera_theta = 0, 0
camera_x, camera_y, camera_z = 4, 4, 4
human_phi, human_theta = 0, 0

print(model.parameters())
betas = torch.tensor([[-0.3596, -1.0232, -1.7584, -2.0465, 0.3387, -0.8562, 0.8869, 0.5013,
                       0.5338, -0.0210]], dtype=torch.float32)
expression = torch.tensor([[2.7228, -1.8139, 0.6270, -0.5565, 0.3251, 0.5643, -1.2158, 1.4149,
                            0.4050, 0.6516]])
output = model(betas=betas, expression=expression,
               return_verts=True)
vertices = output.vertices.detach().cpu().numpy().squeeze()
joints = output.joints.detach().cpu().numpy().squeeze()

print('Vertices shape =', vertices.shape)
print('Joints shape =', joints.shape)

vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]
tri_mesh = trimesh.Trimesh(vertices, model.faces,
                           vertex_colors=vertex_colors)

mesh = pyrender.Mesh.from_trimesh(tri_mesh)

scene = pyrender.Scene()
human_pose = get_pose_matrix(phi=human_phi, theta=human_theta)
scene.add(mesh, pose=human_pose)

# pyrender.Viewer(scene, use_raymond_lighting=True, rotate=False, show_world_axis=True)
camera = pyrender.PerspectiveCamera(yfov=np.pi / 3, aspectRatio=1.0)

phi, theta = camera_origin_direction(camera_x, camera_y, camera_z)
camera_pose = get_pose_matrix(phi=-phi, theta=theta, x=camera_x, y=camera_y, z=camera_z)

print(camera_pose)
scene.add(camera, pose=camera_pose)
r = pyrender.OffscreenRenderer(1000, 1000)
color, depth = r.render(scene)
plt.figure()
plt.axis('off')
plt.imshow(color)
plt.imsave("images/cameraphi_{}humanphi_{}humantheta_{}.png".format(camera_phi,
                                                                    human_phi, human_theta), color)
plt.show()
