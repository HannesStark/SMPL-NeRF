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


class VertexSphereDataset(Dataset):
    """
     Takes a data directory of type "smpl_nerf" and returns rays
    """

    def __init__(self, image_directory: str, transforms_file: str, vertex_radius,
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
        self.vertex_radius = vertex_radius
        self.rays = []  # list of arrays with ray translation, ray direction and rgb
        self.human_poses = []  # list of corresponding human poses
        self.goal_smpls = []
        print('Start initializing all rays of all images')
        with open(transforms_file, 'r') as transforms_file:
            transforms_dict = json.load(transforms_file)
        camera_angle_x = transforms_dict['camera_angle_x']
        image_transform_map = transforms_dict.get('image_transform_map')
        image_pose_map = transforms_dict.get('image_pose_map')
        self.expression = [transforms_dict['expression']]
        self.betas = [transforms_dict['betas']]
        self.canonical_smpl = torch.from_numpy(get_smpl_vertices(self.betas, self.expression))
        image_paths = sorted(glob.glob(os.path.join(image_directory, '*.png')))
        if not len(image_paths) == len(image_transform_map):
            raise ValueError('Number of images in image_directory is not the same as number of transforms')
        for image_path in image_paths:
            camera_transform = np.array(image_transform_map[os.path.basename(image_path)])
            human_pose = torch.tensor(image_pose_map[os.path.basename(image_path)])

            image = cv2.imread(image_path)
            self.h, self.w = image.shape[:2]

            # should we append a list of the different h, w of all images? right now referencing only the last h, w
            self.focal = .5 * self.w / np.tan(.5 * camera_angle_x)
            rays_translation, rays_direction = get_rays(self.h, self.w, self.focal, camera_transform)

            trans_dir_rgb_stack = np.stack([rays_translation, rays_direction, image], -2)
            trans_dir_rgb_list = trans_dir_rgb_stack.reshape((-1, 3, 3))
            self.human_poses.append(np.repeat(human_pose[np.newaxis, :], trans_dir_rgb_list.shape[0], axis=0))
            self.goal_smpls.append(
                torch.from_numpy(get_smpl_vertices(self.betas, self.expression, body_pose=human_pose[None, :])))

            self.rays.append(trans_dir_rgb_list)
        self.rays = np.concatenate(self.rays)
        self.human_poses = np.concatenate(self.human_poses)

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

        rays_translation, rays_direction, rgb = self.rays[index]

        ray_samples, samples_translations, samples_directions, z_vals, rgb = self.transform(
            (rays_translation, rays_direction, rgb))

        goal_smpl = self.goal_smpls[index // (self.h * self.w)]  # divide because we have so many rays per image/smpl
        distances = ray_samples[:, None, :].expand((-1, goal_smpl.shape[0], -1)) - goal_smpl[None, :,
                                                                                   :]  # [number_samples, number_vertices, 3]
        distances = torch.norm(distances, dim=-1, keepdim=True)

        assignments = distances
        mask_to_0 = [assignments > self.vertex_radius]
        mask_to_1 = [assignments < self.vertex_radius]
        assignments[mask_to_0] = 0  # [number_samples, number_vertices, 3]
        assignments[mask_to_1] = 1  # [number_samples, number_vertices, 3]

        warp = self.canonical_smpl - goal_smpl  # [number_vertices,3]
        warp = warp[None, :, :] * assignments  # [number_samples, number_vertices, 3]
        warp = warp.sum(dim=1)  # [number_samples, number_vertices, 3]
        warp = warp / (assignments.sum(dim=1) + 1e-10)  # [number_samples, 3]

        return ray_samples, samples_translations, samples_directions, z_vals, warp.float(), rgb

    def __len__(self) -> int:
        return len(self.rays)
