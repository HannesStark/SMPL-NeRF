from typing import Tuple, Union, List

import numpy as np
import torch
import torch
from torch.utils.data import Dataset


class ToTensor():
    """Add channels dimension if there is none and turn np.array into torch.Tensor."""

    def __init__(self):
        pass

    def __call__(self, segments: Union[Tuple[np.ndarray, np.ndarray, int], Tuple[np.ndarray, np.ndarray]]):
        if len(segments) == 3:  # case of sample rate included in tuple
            noisy_segment, clean_segment, sample_rate = segments
            if noisy_segment.ndim == 1:
                noisy_segment = noisy_segment[np.newaxis, ...]
            if clean_segment.ndim == 1:
                clean_segment = clean_segment[np.newaxis, ...]

            return torch.from_numpy(noisy_segment).float(), torch.from_numpy(clean_segment).float(), sample_rate
        noisy_segment, clean_segment = segments
        if noisy_segment.ndim == 1:
            noisy_segment = noisy_segment[np.newaxis, ...]
        if clean_segment.ndim == 1:
            clean_segment = clean_segment[np.newaxis, ...]

        return torch.from_numpy(noisy_segment).float(), torch.from_numpy(clean_segment).float()


class Normalize():
    """Normalize to [-1,1]."""

    def __init__(self):
        pass

    def __call__(self, ray):
        ray_translation, ray_direction, rgb = ray
        ray_direction = ray_direction / np.linalg.norm(ray_direction, axis=-1, keepdims=True)
        return ray_direction, ray_translation, rgb


class RaySampling():

    def __init__(self, near: int, far: int, number_samples: int):
        self.near = near
        self.far = far
        self.number_samples = number_samples

    def __call__(self, ray):
        ray_translation, ray_direction, rgb = ray
        # get bins along the ray
        t_vals = np.linspace(0., 1., self.number_samples)
        z_vals = 1. / (1. / self.near * (1. - t_vals) + 1. / self.far * (t_vals))
        mids = .5 * (z_vals[1:] + z_vals[:-1])
        upper = np.concatenate([mids, z_vals[-1:]], -1)
        lower = np.concatenate([z_vals[:1], mids], -1)
        # get coarse samples in each bin of the ray
        t_rand = np.random.uniform(z_vals.shape)
        z_vals = lower + (upper - lower) * t_rand
        ray_samples = ray_translation[None, :] + ray_direction[None, :] * z_vals[:, None]  # [N_samples, 3]
        return ray_samples, ray_translation, ray_direction, rgb
