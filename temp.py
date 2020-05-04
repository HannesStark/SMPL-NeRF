import os
import pickle

from torch.utils.data import Subset

import numpy as np
import torch
from torchvision.transforms import transforms
from skimage.color import gray2rgb

from camera import get_circle_pose, get_pose_matrix
from datasets.rays_from_images_dataset import RaysFromImagesDataset
from datasets.transforms import NormalizeRGB, CoarseSampling, ToTensor
from render import get_smpl_mesh, render_scene, save_render

height, width, yfov = 128, 128, np.pi / 3
camera_radius = 2.4
degrees = np.arange(90, 120, 2)

smpl_file_name = "SMPLs/smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl"
texture_file_name = 'textures/texture.jpg'
uv_map_file_name = 'textures/smpl_uv_map.npy'
save_location = 'baseline/pytorch-CycleGAN-and-pix2pix/datasets/testdir/test'
mesh = get_smpl_mesh(smpl_file_name, texture_file_name, uv_map_file_name)

if not os.path.exists(save_location):
    os.makedirs(save_location)

for i, degree in enumerate(degrees):
    camera_pose = get_circle_pose(degree, camera_radius)
    rgb, depth = render_scene(mesh, camera_pose, get_pose_matrix(), camera_pose,
                              height, width, yfov, return_depth=True)

    depth = (depth / (camera_radius * 2) * 255).astype(np.uint8)
    img = np.concatenate([rgb, gray2rgb(depth)], 1)
    save_render(img, os.path.join(save_location, "img_{:03d}.png".format(i)))
