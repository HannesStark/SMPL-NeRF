import os
from typing import Dict

import cv2
import numpy as np
import pickle

import torch
from torchvision.transforms import transforms

from camera import get_sphere_poses
from datasets.rays_from_cameras_dataset import RaysFromCamerasDataset
from datasets.transforms import CoarseSampling, NormalizeRGB, ToTensor
from models.render_ray_net import RenderRayNet
from utils import run_nerf_pipeline, PositionalEncoder, get_rays


def infer(run_file, camera_transforms, output_dir='renders', batch_size=128):
    with open(run_file, 'rb') as file:
        run = pickle.load(file)
    model_coarse = run['model_coarse']
    model_fine = run['model_fine']
    h, w, f = run['height'], run['width'], run['focal']
    rays_dataset = RaysFromCamerasDataset(camera_transforms, h, w, f,
                                          transform=run['dataset_transform'])
    rays_loader = torch.utils.data.DataLoader(rays_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    position_encoder = PositionalEncoder(run['position_encoder']['number_frequencies'],
                                         run['position_encoder']['include_identity'])
    direction_encoder = PositionalEncoder(run['direction_encoder']['number_frequencies'],
                                          run['direction_encoder']['include_identity'])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    model_coarse.to(device)
    model_fine.to(device)
    rgb_images = []
    for i, ray_batch in enumerate(rays_loader):
        ray_samples, ray_translation, ray_direction, z_vals = ray_batch
        ray_samples = ray_samples.to(device)  # [batchsize, number_coarse_samples, 3]
        ray_translation = ray_translation.to(device)  # [batchsize, 3]
        ray_direction = ray_direction.to(device)  # [batchsize, 3]
        z_vals = z_vals.to(device)  # [batchsize, number_coarse_samples]

        _, rgb_fine = run_nerf_pipeline(ray_samples, ray_translation, ray_direction, z_vals,
                                        model_coarse, model_fine, 0,
                                        run['number_fine_samples'], run['white_background'],
                                        position_encoder, direction_encoder)
        rgb_images.append(rgb_fine.detach().cpu().numpy())

    rgb_images = np.concatenate(rgb_images, 0).reshape((len(camera_transforms), h, w, 3))
    if not os.path.exists(output_dir):  # create directory if it does not already exist
        os.mkdir(output_dir)
    for i, image in enumerate(rgb_images):
        basename = os.path.basename(run_file)
        cv2.imwrite(os.path.join(output_dir, os.path.splitext(basename)[0] + '_img_{:03d}.png'.format(i)), image * 255)


if __name__ == '__main__':
    with open('data/val/transforms.pkl', 'rb') as transforms_file:
        transforms_dict = pickle.load(transforms_file)
    image_transform_map: Dict = transforms_dict.get('image_transform_map')
    infer('runs/Apr16_17-24-53_hannes-MS-7721/test.pkl', list(image_transform_map.values()), batch_size=1000)
