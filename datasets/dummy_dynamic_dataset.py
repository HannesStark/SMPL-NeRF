import glob
import os
import json

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils import get_rays


class DummyDynamicDataset(Dataset):
    """
    Dummy dataset that returns indices for the dummy smpl estimator model instead of images. The dummy smpl estimator
    uses the indices to map a ray to the goal pose that is present in the image.
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
        self.image_indices = []
        self.goal_poses = []  # list of corresponding human poses
        print('Start initializing all rays of all images')
        with open(transforms_file, 'r') as transforms_file:
            transforms_dict = json.load(transforms_file)
        camera_angle_x = transforms_dict['camera_angle_x']
        self.image_transform_map = transforms_dict.get('image_transform_map')
        image_pose_map = transforms_dict.get('image_pose_map')
        self.expression = torch.tensor([transforms_dict['expression']])
        self.betas = torch.tensor([transforms_dict['betas']])
        image_paths = sorted(glob.glob(os.path.join(image_directory, '*.png')))
        if not len(image_paths) == len(self.image_transform_map):
            raise ValueError('Number of images in image_directory is not the same as number of transforms')
        for i, image_path in enumerate(image_paths):
            camera_transform = np.array(self.image_transform_map[os.path.basename(image_path)])
            human_pose = np.array(image_pose_map[os.path.basename(image_path)])

            image = cv2.imread(image_path)
            self.h, self.w = image.shape[:2]

            # should we append a list of the different h, w of all images? right now referencing only the last h, w
            self.focal = .5 * self.w / np.tan(.5 * camera_angle_x)
            rays_translation, rays_direction = get_rays(self.h, self.w, self.focal, camera_transform)

            trans_dir_rgb_stack = np.stack([rays_translation, rays_direction, image], -2)
            trans_dir_rgb_list = trans_dir_rgb_stack.reshape((-1, 3, 3))
            self.image_indices.append(torch.ones(trans_dir_rgb_list.shape[0]).long() * i)
            self.goal_poses.append(np.repeat(human_pose[np.newaxis, :], trans_dir_rgb_list.shape[0], axis=0))
            self.rays.append(trans_dir_rgb_list)
        self.rays = np.concatenate(self.rays)
        self.image_indices = torch.cat(self.image_indices)
        self.goal_poses = torch.from_numpy(np.concatenate(self.goal_poses)).float()
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
        """
        rays_translation, rays_direction, rgb = self.rays[index]

        ray_samples, samples_translations, samples_directions, z_vals, rgb = self.transform(
            (rays_translation, rays_direction, rgb))

        return ray_samples, samples_translations, samples_directions, z_vals, self.image_indices[index], rgb

    def __len__(self) -> int:
        return len(self.rays)
