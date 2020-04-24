import os
import pickle

from torch.utils.data import Subset

import numpy as np
import torch
from torchvision.transforms import transforms

from camera import get_circle_pose, get_pose_matrix
from datasets.rays_from_images_dataset import RaysFromImagesDataset
from datasets.transforms import NormalizeRGB, CoarseSampling, ToTensor
from render import get_smpl_mesh, render_scene, save_render

height, width, yfov = 512, 512, np.pi / 3
camera_radius = 2.4
start_angle, end_angle = -20, 20


smpl_file_name = "SMPLs/smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl"
texture_file_name = 'textures/texture.jpg'
uv_map_file_name = 'textures/smpl_uv_map.npy'
mesh = get_smpl_mesh(smpl_file_name, texture_file_name, uv_map_file_name)

for i in range(start_angle, end_angle):
    camera_pose = get_circle_pose(i, camera_radius)
    rgb = render_scene(mesh, camera_pose, get_pose_matrix(), camera_pose,
                       height, width, yfov)
    save_render(rgb, os.path.join('renders', '512_' + str(i) + '.png'))
