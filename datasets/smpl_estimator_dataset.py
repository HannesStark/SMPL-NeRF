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


class SmplEstimatorDataset(Dataset):
    """
    Dataset to train Smpl Estimator supervised.
    Returns (Image, human_pose) - pairs
    """

    def __init__(self, image_directory: str, transforms_file: str, args,
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
        self.args = args
        self.transform = transform
        self.images = []
        self.human_poses = []  # list of corresponding human poses

        print('Start loading images')
        with open(transforms_file, 'r') as transforms_file:
            transforms_dict = json.load(transforms_file)
        self.expression = [transforms_dict['expression']]
        self.betas = [transforms_dict['betas']]
        image_pose_map = transforms_dict.get('image_pose_map')
        image_paths = sorted(glob.glob(os.path.join(image_directory, 'img_*.png')))
        for i, image_path in enumerate(image_paths):
            image = cv2.imread(image_path)
            human_pose = np.array(image_pose_map[os.path.basename(image_paths[i])])
            self.number_human_pose_params = len(human_pose)
            self.h, self.w = image.shape[:2]
            self.images.append(image)
            self.human_poses.append(human_pose)
        self.images = np.concatenate(self.images).reshape(-1, self.h, self.w, 3)
        self.human_poses = np.concatenate(self.human_poses).reshape(-1, 
                                            self.number_human_pose_params)
        self.canonical_smpl = get_smpl_vertices(self.betas, self.expression)

        print('Finish loading images')

    def __getitem__(self, index: int):
        """
        Parameters
        ----------
        index : int
            Index of ray.

        Returns
        -------
        image: torch.Tensor ([h, w, 3])
            RGB-image
        human_pose: torch.Tensor ([69])
            goal pose
        """
        image = self.images[index]
        human_pose = self.human_poses[index]
        image= self.transform((image))
        image = torch.from_numpy(image)
        human_pose = torch.from_numpy(human_pose)
        return image, human_pose

    def __len__(self) -> int:
        return len(self.images)
