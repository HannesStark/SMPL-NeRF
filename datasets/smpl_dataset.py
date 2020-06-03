import glob
import os
import json

import cv2
import numpy as np
import torch
from torch.distributions import MultivariateNormal
from torch.utils.data import Dataset

from utils import get_rays
import smplx
from render import get_smpl_vertices
import torch.distributions as D


class SmplDataset(Dataset):
    """
    Dataset of rays from a directory of images and an images camera transforms
    mapping file (used for training).
    """

    def __init__(self, image_directory: str, transforms_file: str,
                 transform) -> None:
        """
        Parameters
        ----------
        image_directory : str
            Path to images.
        transforms_file : str
            File path to file containing transformation mappings.
        transform :
            List of callable transforms for preprocessing.
        """
        super().__init__()
        self.transform = transform
        self.rays = []  # list of arrays with ray translation, ray direction and rgb
        self.human_poses = []  # list of corresponding human poses
        self.depth = []
        self.warp = []

        print('Start initializing all rays of all images')
        with open(transforms_file, 'r') as transforms_file:
            transforms_dict = json.load(transforms_file)
        camera_angle_x = transforms_dict['camera_angle_x']
        image_transform_map = transforms_dict.get('image_transform_map')
        image_pose_map = transforms_dict.get('image_pose_map')
        self.expression = [transforms_dict['expression']]
        self.betas = [transforms_dict['betas']]
        image_paths = sorted(glob.glob(os.path.join(image_directory, 'img_*.png')))
        depth_paths = sorted(glob.glob(os.path.join(image_directory, 'depth_*.npy')))
        warp_paths = sorted(glob.glob(os.path.join(image_directory, 'warp_*.npy')))

        if not len(image_paths) == len(image_transform_map):
            raise ValueError('Number of images in image_directory is not the same as number of transforms')

        for i in range(len(image_transform_map)):
            camera_transform = np.array(image_transform_map[os.path.basename(image_paths[i])])
            human_pose = np.array(image_pose_map[os.path.basename(image_paths[i])])

            image = cv2.imread(image_paths[i])
            depth = np.load(depth_paths[i])
            warp = np.load(warp_paths[i])

            self.h, self.w = image.shape[:2]

            self.focal = .5 * self.w / np.tan(.5 * camera_angle_x)
            rays_translation, rays_direction = get_rays(self.h, self.w, self.focal, camera_transform)

            trans_dir_rgb_stack = np.stack([rays_translation, rays_direction, image], -2)  # [h x w x 3 x 3]
            trans_dir_rgb_list = trans_dir_rgb_stack.reshape((-1, 3, 3))
            self.human_poses.append(np.repeat(human_pose[np.newaxis, :], trans_dir_rgb_list.shape[0], axis=0))
            self.rays.append(trans_dir_rgb_list)

            self.depth.append(depth.reshape((trans_dir_rgb_list.shape[0], 1)))
            self.warp.append(warp.reshape((trans_dir_rgb_list.shape[0], 3)))

        self.rays = np.concatenate(self.rays)
        self.human_poses = np.concatenate(self.human_poses)
        self.warp = np.concatenate(self.warp)
        self.depth = np.concatenate(self.depth)
        self.canonical_smpl = get_smpl_vertices(self.betas, self.expression)

        print('Finish initializing rays')

    def __getitem__(self, index: int):
        """
        Parameters
        ----------
        index : int
            Index of ray.

        Returns
        -------
        ray_samples : torch.Tensor ([number_coarse_samples, 3])
            Coarse samples along the ray between near and far bound.
        samples_translations : torch.Tensor ([3])
            Translation of samples.
        samples_directions : torch.Tensor ([3])
            Direction of samples.
        z_vals : torch.Tensor ([number_coarse_samples])
            Depth of coarse samples along ray.
        rgb : torch.Tensor ([3])
            RGB value corresponding to ray.
        human_pose: torch.Tensor ([69])
            goal pose
        warp: torch.Tensor ([3])
            warp from goal to canonical for 3D sample
        depth: torch.Tensor ([1])
            depth value for sample on ray
        """

        rays_translation, rays_direction, rgb = self.rays[index]

        ray_sample, samples_translation, samples_direction, z_vals, rgb = self.transform(
            (rays_translation, rays_direction, rgb))
        return ray_sample, samples_translation, samples_direction, z_vals, torch.Tensor(
            self.human_poses[index]).float(), torch.Tensor(self.warp[index]).float(), torch.Tensor(
            self.depth[index]).float(), rgb

    def __len__(self) -> int:
        return len(self.rays)
