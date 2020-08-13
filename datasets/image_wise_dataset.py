import glob
import os
import json

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils import get_rays


class ImageWiseDataset(Dataset):
    """
    Dummy dataset that returns indices for the dummy smpl estimator model instead of images. The dummy smpl estimator
    uses the indices to map a ray to the goal pose that is present in the image.
    """

    def __init__(self, image_directory: str, transforms_file: str,
                 transform, args) -> None:
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
        self.number_samples = args.number_coarse_samples
        self.near = args.near
        self.far = args.far

        with open(transforms_file, 'r') as transforms_file:
            transforms_dict = json.load(transforms_file)
        self.camera_angle_x = transforms_dict['camera_angle_x']
        self.image_transform_map = transforms_dict.get('image_transform_map')
        self.image_paths = sorted(glob.glob(os.path.join(image_directory, '*.png')))
        if not len(self.image_paths) == len(self.image_transform_map):
            raise ValueError('Number of images in image_directory is not the same as number of transforms')

        self.goal_poses = []  # list of corresponding human pose
        image_pose_map = transforms_dict.get('image_pose_map')
        self.expression = torch.tensor([transforms_dict['expression']])
        self.betas = torch.tensor([transforms_dict['betas']])
        for i, image_path in enumerate(self.image_paths):
            human_pose = np.array(image_pose_map[os.path.basename(image_path)])
            self.goal_poses.append(human_pose[np.newaxis, :])
            image = cv2.imread(image_path)
            self.h, self.w = image.shape[:2]

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
        image_path = self.image_paths[index]
        camera_transform = np.array(self.image_transform_map[os.path.basename(image_path)])

        image = cv2.imread(image_path)
        self.h, self.w = image.shape[:2]

        # should we append a list of the different h, w of all images? right now referencing only the last h, w
        self.focal = .5 * self.w / np.tan(.5 * self.camera_angle_x)
        rays_translation, rays_direction = get_rays(self.h, self.w, self.focal, camera_transform)

        rays_translation = rays_translation.reshape(-1,3)
        rays_direction = rays_direction.reshape(-1,3)
        rgb = image.reshape(-1,3)

        t_vals = np.linspace(0., 1., self.number_samples)
        z_vals = 1. / (1. / self.near * (1. - t_vals) + 1. / self.far * (t_vals))
        mids = .5 * (z_vals[1:] + z_vals[:-1])
        upper = np.concatenate([mids, z_vals[-1:]], -1)
        lower = np.concatenate([z_vals[:1], mids], -1)
        # get coarse samples in each bin of the ray
        z_vals = lower + (upper - lower) * np.random.rand()
        rays_samples = rays_translation[:, None, :] + rays_direction[:, None, :] * np.repeat(z_vals[None, :, None], len(rgb), axis=0)  # [N_samples, 3]

        rays_samples = torch.from_numpy(rays_samples).float()
        rays_translation = torch.from_numpy(rays_translation).float()
        rays_direction = torch.from_numpy(rays_direction).float()
        z_vals = torch.from_numpy(z_vals).float()
        rgb = (np.array(rgb) / 255.).astype(np.float32)
        rgb = torch.from_numpy(rgb).float()

        return rays_samples, rays_translation, rays_direction, z_vals, rgb

    def __len__(self) -> int:
        return len(self.image_paths)
