import numpy as np
import torch


class ToTensor():
    """
    Turn np.array into torch.Tensor.
    """

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


class NormalizeRGB():
    """
    Normalize RGB image to [0, 1]
    """
    def __init__(self):
        pass

    def __call__(self, ray):
        ray_translation, ray_direction, rgb = ray
        rgb = (np.array(rgb) / 255.).astype(np.float32)
        return ray_translation, ray_direction, rgb


class CoarseSampling():
    """
    Coarse sampling along a ray
    """
    def __init__(self, near: int, far: int, number_samples: int = 64):
        """
        Parameters
        ----------
        near : int
            Near bound for coarse sampling.
        far : int
            Far bound for coarse sampling.
        number_samples : int, optional
            Number of coarse samples along the ray. The default is 64.
        """
        self.near = near
        self.far = far
        self.number_samples = number_samples

    def __call__(self, ray):
        """
        Performs coars sampling on ray

        Parameters
        ----------
        ray : Tuple
            Return of get_rays() and rgb value for ray.

        Returns
        -------
        ray_samples : np.array (number_samples, 3)
            Coarse samples along the ray between near and far bound.
        ray_translation : np.array (3, )
            Translation of ray.
        ray_direction : np.array (3, )
            Direction of ray.
        z_vals : np.array (64, )
            Depth of coarse samples along ray.
        rgb : np.array (3, )
            RGB values corresponding to ray.
        """
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
        return ray_samples, ray_translation, ray_direction, z_vals, rgb
