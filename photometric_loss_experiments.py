#!/usr/bin/env python3
from render import render_scene, get_smpl_mesh, get_human_poses
from camera import get_pose_matrix, get_sphere_pose
import numpy as np
import matplotlib.pyplot as plt
import smplx
import torch
from PIL import Image
from io import BytesIO


mesh_canonical = get_smpl_mesh()
human_poses = get_human_poses([38, 41], -80, 80, 31)
height = 128
width = 128
camera_angle_x = np.pi/3

camera_pose = get_sphere_pose(0, 0, 2.4)
img_canonical = render_scene(mesh_canonical, camera_pose, get_pose_matrix(), camera_pose,
                               height, width, camera_angle_x)/255.0
num_pixels = img_canonical.reshape((-1)).shape[0]
photometric_losses = []
for human_pose in human_poses:
    mesh_goal = get_smpl_mesh(body_pose = human_pose)
    img_goal = render_scene(mesh_goal, camera_pose, get_pose_matrix(), camera_pose,
                               height, width, camera_angle_x)/255.0
    #plt.imshow(img_goal)
    photometric_loss = 1/num_pixels*np.sum(np.abs(img_canonical - img_goal))
    photometric_losses.append(photometric_loss)
plt.plot(photometric_losses)
plt.show()

