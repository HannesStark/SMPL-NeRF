import glob
import os
import json

import cv2
import numpy as np
import torch
from torch.distributions import MixtureSameFamily
from torch.utils.data import Dataset
from trimesh.ray.ray_triangle import RayMeshIntersector
import torch.distributions as D

from render import get_smpl_mesh
from utils import get_rays


class ImageWiseDataset(Dataset):
    """
    Dummy dataset that returns indices for the dummy smpl estimator model instead of images. The dummy smpl estimator
    uses the indices to map a ray to the goal pose that is present in the image.
    """

    def __init__(self, image_directory: str, transforms_file: str, goal_pose,
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
        self.args = args
        self.goal_pose = goal_pose

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
        rays_direction = rays_direction.reshape(-1, 3)
        rays_translation = rays_translation.reshape(-1, 3)

        t_vals = np.linspace(0., 1., self.args.number_coarse_samples)
        z_vals = 1. / (1. / self.args.near * (1. - t_vals) + 1. / self.args.far * (t_vals))
        mids = .5 * (z_vals[1:] + z_vals[:-1])
        upper = np.concatenate([mids, z_vals[-1:]], -1)
        lower = np.concatenate([z_vals[:1], mids], -1)
        # get coarse samples in each bin of the ray
        z_vals_simple = torch.from_numpy(lower + (upper - lower) * np.random.rand())

        goal_mesh = get_smpl_mesh(body_pose=self.goal_pose, return_pyrender=False)
        intersector = RayMeshIntersector(goal_mesh)
        z_vals_image = []
        for ray_index in range(len(rays_translation)):
            intersections = intersector.intersects_location([rays_translation[ray_index]],
                                                            [rays_direction[ray_index]])
            canonical_intersections_points = torch.from_numpy(intersections[0])  # (N_intersects, 3)
            if self.args.number_coarse_samples == 1:
                if len(canonical_intersections_points) == 0:
                    z_vals = torch.DoubleTensor([self.args.far])  # [1]
                else:
                    distances_camera = torch.norm(canonical_intersections_points - rays_translation[ray_index],
                                                  dim=1)
                    z_vals = torch.tensor([torch.min(distances_camera)])  # [1]
            elif self.args.coarse_samples_from_intersect == 1:
                if len(canonical_intersections_points) == 0:
                    z_vals = z_vals_simple
                else:
                    distances_camera = torch.norm(canonical_intersections_points - rays_translation[ray_index],
                                                  dim=1)
                    mean = torch.min(distances_camera)
                    gauss = D.Normal(mean,
                                     torch.ones_like(mean) * self.args.std_dev_coarse_sample_prior)
                    z_vals, _ = torch.sort(gauss.sample((self.args.number_coarse_samples,)))
            elif len(canonical_intersections_points) == 0 or self.args.coarse_samples_from_prior != 1:
                z_vals = z_vals_simple
            else:
                mix = D.Categorical(torch.ones(len(canonical_intersections_points), ))
                means = torch.norm(canonical_intersections_points - rays_translation[ray_index], dim=-1)
                comp = D.Normal(means, torch.ones_like(means) * self.args.std_dev_coarse_sample_prior)
                gmm = MixtureSameFamily(mix, comp)
                z_vals = gmm.sample((self.args.number_coarse_samples,))
            z_vals_image.append(z_vals)
        z_vals_image = torch.stack(z_vals_image)  # [h*w, number_coarse_samples]
        if self.args.number_coarse_samples == 1:
            z_vals_image = z_vals_image.view(-1, 1)
        rays_translation = torch.from_numpy(rays_translation)
        rays_direction = torch.from_numpy(rays_direction)
        rays_samples = rays_translation[:, None, :] + rays_direction[:, None, :] * z_vals_image[:, :,
                                                                                   None]  # [h*w, number_coarse_samples, 3]
        rays_translation = rays_translation.reshape(-1, 3)
        rays_direction = rays_direction.reshape(-1, 3)
        rgb = image.reshape(-1, 3)

        rgb = (np.array(rgb) / 255.).astype(np.float32)
        rgb = torch.from_numpy(rgb).float()

        return rays_samples.float(), rays_translation.float(), rays_direction.float(), z_vals.float(), rgb

    def __len__(self) -> int:
        return len(self.image_paths)
