#!/usr/bin/env python3
from render import render_scene, get_smpl_mesh, get_human_poses
from camera import get_pose_matrix, get_sphere_pose
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import smplx
import torch
from PIL import Image
from io import BytesIO


mesh_canonical = get_smpl_mesh()

human_poses = get_human_poses([38, 41], -60, 60, 31)
height = 128
width = 128
camera_angle_x = np.pi/3

camera_pose = get_sphere_pose(0, 0, 2.4)
img_canonical = render_scene(mesh_canonical, camera_pose, get_pose_matrix(), camera_pose,
                               height, width, camera_angle_x)/255.0
num_pixels = img_canonical.reshape((-1)).shape[0]
photo_l1_losses = []
photo_l2_losses = []
photo_huber_losses = []

for human_pose in human_poses:
    mesh_goal = get_smpl_mesh(body_pose = human_pose)
    img_goal = render_scene(mesh_goal, camera_pose, get_pose_matrix(), camera_pose,
                               height, width, camera_angle_x)/255.0
    #plt.imshow(img_goal)
    photo_loss_1 = 1/num_pixels * np.sum(np.abs(img_canonical - img_goal))
    photo_loss_2 = 1/num_pixels * np.linalg.norm(img_canonical - img_goal)
    photo_loss_huber = 1/num_pixels * np.sum(sc.special.huber(1, img_canonical - img_goal))

    photo_l1_losses.append(photo_loss_1)
    photo_l2_losses.append(photo_loss_2)
    photo_huber_losses.append(photo_loss_huber)

plt.plot(np.linspace(-60,60,31), photo_l1_losses)
plt.title("L1")
plt.show()

plt.plot(np.linspace(-60,60,31), photo_l2_losses)
plt.title("L2")
plt.show()

plt.plot(np.linspace(-60,60,31), photo_huber_losses)
plt.title("Huber Loss")
plt.show()



