import os

import cv2
import numpy as np
import pickle

import torch
from torchvision.transforms import transforms

from datasets.rays_from_cameras_dataset import RaysFromCamerasDataset
from datasets.transforms import CoarseSampling, NormalizeRGB, ToTensor
from models.render_ray_net import RenderRayNet
from utils import run_nerf_pipeline, PositionalEncoder, get_rays


def infer(run_name, camera_transforms, output_dir='renders', batch_size=128):
    with open(run_name + '.pkl', 'rb') as file:
        run = pickle.load(file)
    model_coarse = run['model_coarse']
    model_fine = run['model_fine']
    h, w, f = run['height'], run['width'], run['focal']
    rays_dataset = RaysFromCamerasDataset(camera_transforms, h, w, f,
                                          transform=run['dataset_transform'])
    rays_loader = torch.utils.data.DataLoader(rays_dataset, batch_size=batch_size, shuffle=False, num_workers=0, )
    position_encoder = PositionalEncoder(run['position_encoder']['number_frequencies'],
                                         run['position_encoder']['include_identity'])
    direction_encoder = PositionalEncoder(run['direction_encoder']['number_frequencies'],
                                          run['direction_encoder']['include_identity'])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    # if torch.cuda.is_available():
    #    torch.set_default_tensor_type('torch.cuda.FloatTensor')
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
        rgb_images.append(rgb_fine)

    rgb_images = torch.cat(rgb_images, 0).view(len(camera_transforms), h, w, 3).detach().numpy()
    if not os.path.exists(output_dir):  # create directory if it does not already exist
        os.mkdir(output_dir)
    for i, image in enumerate(rgb_images):
        print(image)
        cv2.imwrite(os.path.join(output_dir, run_name + '_img_{:03d}.png'.format(i)), image*255)


if __name__ == '__main__':
    infer('first_experiment', [np.array([[
        -0.9999021887779236,
        0.004192245192825794,
        -0.013345719315111637,
        -0.05379832163453102
    ],
        [
            -0.013988681137561798,
            -0.2996590733528137,
            0.95394366979599,
            3.845470428466797
        ],
        [
            -4.656612873077393e-10,
            0.9540371894836426,
            0.29968830943107605,
            1.2080823183059692
        ],
        [
            0.0,
            0.0,
            0.0,
            1.0
        ]])], batch_size=1000)
