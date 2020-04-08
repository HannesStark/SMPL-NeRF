import numpy as np
import torch


class ToTensor():
    """Add channels dimension if there is none and turn np.array into torch.Tensor."""

    def __init__(self):
        pass

    def __call__(self, ray):
        ray_samples, samples_translations, samples_directions, z_vals, rgb = ray
        ray_samples = torch.from_numpy(ray_samples).float()
        samples_translations = torch.Tensor(samples_translations).float()
        samples_directions = torch.Tensor(samples_directions).float()
        z_vals = torch.from_numpy(z_vals).float()
        rgb = torch.from_numpy(rgb).float()

        return ray_samples, samples_translations, samples_directions, z_vals, rgb


class CoarseSampling():

    def __init__(self, near: int, far: int, number_samples: int = 64):
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
        z_vals = lower + (upper - lower) * np.random.rand()
        ray_samples = ray_translation[None, :] + ray_direction[None, :] * z_vals[:, None]  # [N_samples, 3]

        samples_directions = [ray_direction] * self.number_samples
        samples_translations = [ray_translation] * self.number_samples
        return ray_samples, samples_translations, samples_directions, z_vals, rgb
