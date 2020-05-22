import os
from typing import Dict

import cv2
import imageio
import numpy as np
import json
from tqdm import tqdm

import torch

from camera import get_circle_pose, get_sphere_pose
from datasets.rays_from_cameras_dataset import RaysFromCamerasDataset
from utils import run_nerf_pipeline, PositionalEncoder


def inference(run_file, camera_transforms, batch_size=128):
    with open(run_file, 'r') as file:
        run = json.load(file)
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
    for i, ray_batch in enumerate(tqdm(rays_loader)):
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
    rgb_images = np.clip(rgb_images, 0, 1) * 255

    return rgb_images.astype(np.uint8)


def save_rerenders(rgb_images, run_file, output_dir='renders'):
    basename = os.path.basename(run_file)
    output_dir = os.path.join(output_dir, os.path.splitext(basename)[0])
    if not os.path.exists(output_dir):  # create directory if it does not already exist
        os.makedirs(output_dir)
    for i, image in enumerate(rgb_images):
        cv2.imwrite(os.path.join(output_dir, 'img_{:03d}.png'.format(i)), image)
    imageio.mimwrite(os.path.join(output_dir, 'animated.mp4'), rgb_images,
                     fps=30, quality=8)


if __name__ == '__main__':
    run_file = 'runs/Apr16_17-24-53_hannes-MS-7721/128_-90_90_100.pkl'
    # Option 1: Use transforms.pkl
    # with open('data/val/transforms.pkl', 'rb') as transforms_file:
    #   transforms_dict = pickle.load(transforms_file)
    # image_transform_map: Dict = transforms_dict.get('image_transform_map')
    # transforms_list = list(image_transform_map.values())
    #
    # Option 2: Use get_circle_pose to create path on circle
    # degrees = [0, 10]
    # transforms_list = []
    # height, width, yfov = 128, 128, np.pi / 3
    # camera_radius = 2.4
    # for i in degrees:
    #    camera_pose = get_circle_pose(i, camera_radius)
    #    transforms_list.append(camera_pose)
    #
    # Option 3: Use circle path on sphere
    angles = np.linspace(0, np.pi*2, 25)
    radius = 45
    transforms_list = []
    height, width, yfov = 512, 512, np.pi / 3
    camera_radius = 2.4
    for angle in angles:
        phi = radius*np.cos(angle)
        theta = radius*np.sin(angle)
        camera_pose = get_sphere_pose(phi, theta, camera_radius)
        transforms_list.append(camera_pose)
    rgb_images = inference(run_file, transforms_list, batch_size=900)
    save_rerenders(rgb_images, run_file)
