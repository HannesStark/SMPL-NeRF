#!/usr/bin/env python3
from render import render_scene, get_smpl_mesh, get_human_poses, get_smpl_mesh_distorted
from camera import get_pose_matrix, get_sphere_pose
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import smplx
import torch
from PIL import Image
from io import BytesIO
import tqdm


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

"""
plt.plot(np.linspace(-60,60,31), photo_l1_losses)
plt.title("L1")
plt.show()
"""

# print 3 losses in one plot row
fig, axs = plt.subplots(1, 3)
axs[0].plot(np.linspace(-60,60,31), photo_l1_losses)
axs[0].set_title("L1")
axs[1].plot(np.linspace(-60,60,31), photo_l2_losses)
axs[1].set_title("L2")
axs[2].plot(np.linspace(-60,60,31), photo_huber_losses)
axs[2].set_title("Huber Loss")
fig.suptitle('Varying Pose Parameter: Arm Angles [-60:60]')
plt.show()

###### vary smpl parameter by linear interpolation of specific beta #######

mesh_undistorted = get_smpl_mesh()

beta = np.linspace(-0.3596-2, -0.3596 + 2, 50)
variances = np.linspace(0, 2, 100)

photo_l1_losses = []
photo_l2_losses = []
photo_huber_losses = []

for b in beta:
    mesh_goal = get_smpl_mesh_distorted(beta=b)
    img_goal = render_scene(mesh_goal, camera_pose, get_pose_matrix(), camera_pose,
                               height, width, camera_angle_x)/255.0
    #plt.imshow(img_goal)
    photo_loss_1 = 1/num_pixels * np.sum(np.abs(img_canonical - img_goal))
    photo_loss_2 = 1/num_pixels * np.linalg.norm(img_canonical - img_goal)
    photo_loss_huber = 1/num_pixels * np.sum(sc.special.huber(1, img_canonical - img_goal))

    photo_l1_losses.append(photo_loss_1)
    photo_l2_losses.append(photo_loss_2)
    photo_huber_losses.append(photo_loss_huber)

fig, axs = plt.subplots(1, 3)
axs[0].plot(beta, photo_l1_losses)
axs[0].set_title("L1")
axs[1].plot(beta, photo_l2_losses)
axs[1].set_title("L2")
axs[2].plot(beta, photo_huber_losses)
axs[2].set_title("Huber Loss")
fig.suptitle('Varying variance of Gaussian Noise')
plt.show()


###### vary smpl parameter by adding Gaussian Noise on all SMPL Parameter #######

variances = np.linspace(0, 2, 100)

photo_l1_losses = []
photo_l2_losses = []
photo_huber_losses = []

for var in tqdm(variances):

    temp_l1 = []
    temp_l2 = []
    temp_lH = []
    for i in range(100):

        mesh_goal = get_smpl_mesh_distorted(var=var)
        img_goal = render_scene(mesh_goal, camera_pose, get_pose_matrix(), camera_pose,
                                   height, width, camera_angle_x)/255.0
        #plt.imshow(img_goal)
        photo_loss_1 = 1/num_pixels * np.sum(np.abs(img_canonical - img_goal))
        photo_loss_2 = 1/num_pixels * np.linalg.norm(img_canonical - img_goal)
        photo_loss_huber = 1/num_pixels * np.sum(sc.special.huber(1, img_canonical - img_goal))

        temp_l1.append(photo_loss_1)
        temp_l2.append(photo_loss_2)
        temp_lH.append(photo_loss_2)

    photo_l1_losses.append(np.mean(temp_l1))
    photo_l2_losses.append(np.mean(temp_l2))
    photo_huber_losses.append(np.mean(temp_lH))

fig, axs = plt.subplots(1, 3)
axs[0].plot(variances, photo_l1_losses)
axs[0].set_title("L1")
axs[1].plot(variances, photo_l2_losses)
axs[1].set_title("L2")
axs[2].plot(variances, photo_huber_losses)
axs[2].set_title("Huber Loss")
fig.suptitle('Varying variance of Gaussian Noise')
plt.show()

def printLosses(x, l1, l2, l3):
    fig, axs = plt.subplots(1, 3)
    axs[0].plot(x, l1)
    axs[0].set_title("L1")
    axs[1].plot(x, l2)
    axs[1].set_title("L2")
    axs[2].plot(x, l3)
    axs[2].set_title("Huber Loss")
    fig.suptitle('Varying variance of Gaussian Noise')
    plt.show()




