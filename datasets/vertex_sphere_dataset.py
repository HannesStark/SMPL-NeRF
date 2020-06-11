import glob
import os
import json
import sys

import cv2
import numpy as np
import torch
from torch.distributions import MultivariateNormal
from torch.utils.data import Dataset

from utils import get_rays
import smplx
from render import get_smpl_vertices
import torch.distributions as D
from tqdm import tqdm


class VertexSphereDataset(Dataset):
    """
     Takes a data directory of type "smpl_nerf" and returns rays
    """

    def __init__(self, image_directory: str, transforms_file: str, args) -> None:
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
        print('Start initializing all rays of all images')
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        with open(transforms_file, 'r') as transforms_file:
            transforms_dict = json.load(transforms_file)
        camera_angle_x = transforms_dict['camera_angle_x']
        image_transform_map = transforms_dict.get('image_transform_map')
        image_pose_map = transforms_dict.get('image_pose_map')
        self.expression = [transforms_dict['expression']]
        self.betas = [transforms_dict['betas']]
        canonical_smpl = torch.from_numpy(get_smpl_vertices(self.betas, self.expression)).to(device)
        image_paths = sorted(glob.glob(os.path.join(image_directory, '*.png')))
        if not len(image_paths) == len(image_transform_map):
            raise ValueError('Number of images in image_directory is not the same as number of transforms')

        # get bins along the ray
        t_vals = np.linspace(0., 1., args.number_coarse_samples)
        z_vals = 1. / (1. / args.near * (1. - t_vals) + 1. / args.far * (t_vals))
        mids = .5 * (z_vals[1:] + z_vals[:-1])
        upper = np.concatenate([mids, z_vals[-1:]], -1)
        lower = np.concatenate([z_vals[:1], mids], -1)
        # get coarse samples in each bin of the ray
        self.z_vals = torch.from_numpy(lower + (upper - lower) * np.random.rand())

        self.rays_samples = []  # list of arrays with ray translation, ray direction and rgb
        self.rays = []
        self.all_warps = []
        for image_path in tqdm(image_paths, desc='Images', leave=False):
            camera_transform = np.array(image_transform_map[os.path.basename(image_path)])
            human_pose = torch.tensor(image_pose_map[os.path.basename(image_path)])

            image = cv2.imread(image_path)
            image.flags.writeable = True
            self.h, self.w = image.shape[:2]
            image = (torch.tensor(image).double() / 255.).view(-1, 3)

            # should we append a list of the different h, w of all images? right now referencing only the last h, w
            self.focal = .5 * self.w / np.tan(.5 * camera_angle_x)
            rays_translation, rays_direction = get_rays(self.h, self.w, self.focal, camera_transform)
            rays_translation = torch.from_numpy(np.copy(rays_translation)).view(-1, 3) # copy because array not writable
            rays_direction = torch.from_numpy(rays_direction).view(-1, 3)
            translation_direction_rgb_stack = torch.stack(
                [rays_translation, rays_direction, image], -2)

            rays_samples = rays_translation[:, None, :] + rays_direction[:, None, :] * self.z_vals[None, :,
                                                                                       None]  # [h*w, number_coarse_samples, 3]

            goal_smpl = torch.from_numpy(get_smpl_vertices(self.betas, self.expression, body_pose=human_pose[None, :]))

            warps_of_image = []

            rays_samples = rays_samples.to(device)
            goal_smpl = goal_smpl.to(device)
            # iterate through all the samples because we do not have enough memeory to compute all warps at once
            for sample_index in tqdm(range(args.number_coarse_samples), desc='Samples'):
                sample = rays_samples[:, sample_index, :]  # [h*w, 3]
                distances = sample[:, None, :].expand((-1, goal_smpl.shape[0], -1)) - goal_smpl[None, :,
                                                                                      :]  # [h*w, number_vertices, 3]
                distances = torch.norm(distances, dim=-1, keepdim=True)
                assignments = distances
                mask_to_0 = [assignments > args.vertex_sphere_radius]
                mask_to_1 = [assignments < args.vertex_sphere_radius]
                assignments[mask_to_0] = 0  # [h*w, number_vertices, 3]
                assignments[mask_to_1] = 1  # [h*w, number_vertices, 3]

                warp = canonical_smpl - goal_smpl  # [h*w, number_vertices, 3]
                warp = warp[None, :, :] * assignments  # [h*w, number_vertices, 3]
                warp = warp.sum(dim=1)  # [h*w, number_vertices, 3]
                warp = warp / (assignments.sum(dim=1) + 1e-10)  # [h*w, 3]
                warps_of_image.append(warp)
            warps_of_image = torch.stack(warps_of_image, -2).cpu()  # [h*w, number_samples, 3]
            rays_samples = rays_samples.cpu()

            self.rays_samples.append(rays_samples)
            self.all_warps.append(warps_of_image)
            self.rays.append(translation_direction_rgb_stack)
        self.rays_samples = torch.cat(self.rays_samples)
        self.all_warps = torch.cat(self.all_warps)
        self.rays = torch.cat(self.rays)

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
        warp : torch.Tensor ([number_coarse_samples, 3])
            The correct warp for every sample as calculated from canonical and goal smpl and args.vertex_sphere_radius.
        rgb : torch.Tensor ([3])
            RGB value corresponding to ray.

        """

        ray_translation, ray_direction, rgb = self.rays[index]

        return self.rays_samples[index].float(), ray_translation.float(), ray_direction.float(), self.z_vals.float(), \
               self.all_warps[index].float(), rgb.float()

    def __len__(self) -> int:
        return len(self.rays)
