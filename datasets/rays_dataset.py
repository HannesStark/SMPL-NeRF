import os
import pickle
from typing import List, Tuple

import cv2
import librosa
import numpy as np
from torch.utils.data import Dataset
import torch

from utils import get_rays


class RaysDataset(Dataset):
    """Dataset of rays from a directory of images and an images camera transforms mapping file.
    """

    def __init__(self, image_directory: str, transforms_file: str, transform) -> None:
        """
        Args:
            audio_dir (string): Path to .wav, .mp3, .flac and other files of audios.
            sample_rate (int): SR to which the audio will be resampled. Using native SR if this is None.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super().__init__()
        self.transform = transform
        self.rays = []  # list of arrays with ray translation, ray direction and rgb
        print('Start initializing all rays of all images')
        with open(transforms_file, 'rb') as transforms_file:
            transforms_dict = pickle.load(transforms_file)
        camera_angle_x = transforms_dict['camera_angle_x']
        image_transform_map = transforms_dict.get('image_transform_map')
        image_names = os.listdir(image_directory)
        if not len(image_names) == len(image_transform_map):
            raise ValueError('Number of images in image_directory is not the same as number of transforms')
        for image_name in image_names:
            image = cv2.imread(os.path.join(image_directory, image_name))
            h, w = image.shape[:2]
            focal = .5 * w / np.tan(.5 * camera_angle_x)
            rays_translation, rays_direction = get_rays(h, w, focal, image_transform_map[image_name])
            trans_dir_rgb_stack = np.stack([rays_translation, rays_direction, image], -2)
            trans_dir_rgb_list = trans_dir_rgb_stack.reshape((-1, 3, 3))
            self.rays.append(trans_dir_rgb_list)
        self.rays = np.concatenate(self.rays)
        print('Finish initializing rays')

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rays_translation, rays_direction, rgb = self.rays[index]
        translation_list, direction_list, rgb_list = self.transform((rays_translation, rays_direction, rgb))

        return translation_list, direction_list, rgb_list

    def __len__(self) -> int:
        return len(self.rays)
