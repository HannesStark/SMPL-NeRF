import os
import pickle

import cv2
import numpy as np
from torch.utils.data import Dataset

from utils import get_rays


class RaysFromCamerasDataset(Dataset):
    """Dataset of rays from a directory of images and an images camera transforms mapping file.
    """

    def __init__(self, camera_transforms, height, widht, focal, transform) -> None:
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
        for transform in camera_transforms:
            rays_translation, rays_direction = get_rays(height, widht, focal, transform)
            trans_dir_stack = np.stack([rays_translation, rays_direction], -2)
            trans_dir_list = trans_dir_stack.reshape((-1, 2, 3))
            self.rays.append(trans_dir_list)
        self.rays = np.concatenate(self.rays)
        print('Finish initializing rays')

    def __getitem__(self, index: int):
        rays_translation, rays_direction = self.rays[index]
        ray_samples, samples_translations, samples_directions, z_vals, _ = self.transform(
            (rays_translation, rays_direction, []))

        return ray_samples, samples_translations, samples_directions, z_vals

    def __len__(self) -> int:
        return len(self.rays)
