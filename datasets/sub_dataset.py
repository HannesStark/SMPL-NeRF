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


class SubDataset(Dataset):
    """
    Dataset used by image wise solver to iterate through the rays in one image in batches
    """

    def __init__(self, rays_samples, rays_translation, rays_direction, rgb) -> None:
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
        self.rays_direction = rays_direction
        self.rays_samples = rays_samples
        self.rays_translation = rays_translation
        self.rgb = rgb

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
        rgb: torch.Tensor ([3])
            rgb value of the pixel that corresponds to that ray.


        # dependency_rays{Ray_samples [samples, 3],ray_trans[3], ray_direction[3],
        # z_vals[samples], ray_w[1], ray_h[1]} x [Number_of_dependent_rays],
        # goal_pose[69]
        """

        return self.rays_samples[index], self.rays_translation[index], self.rays_direction[index], self.rgb[index]

    def __len__(self) -> int:
        return len(self.rays_samples)
