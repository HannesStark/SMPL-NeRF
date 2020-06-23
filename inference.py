import os
import cv2
import imageio
import numpy as np
import json
import matplotlib.pyplot as plt
import torch

from torchvision.transforms import transforms

import configargparse
from config_parser import config_parser
from models.append_to_nerf_pipeline import AppendToNerfPipeline

from models.render_ray_net import RenderRayNet
from models.warp_field_net import WarpFieldNet
from models.smpl_pipeline import SmplPipeline
from models.smpl_nerf_pipeline import SmplNerfPipeline
from models.nerf_pipeline import NerfPipeline

from datasets.smpl_nerf_dataset import SmplNerfDataset
from datasets.rays_from_images_dataset import RaysFromImagesDataset
from datasets.smpl_dataset import SmplDataset
from datasets.transforms import CoarseSampling, ToTensor, NormalizeRGB

from utils import PositionalEncoder

def inference():
    parser_inference = config_parser_inference()
    args_inference = parser_inference.parse_args()
    parser_training = config_parser()
    config_file_training = os.path.join(args_inference.run_dir, "config.txt")
    parser_training.add_argument('--config2', is_config_file=True, 
                     default=config_file_training, help='config file path')
    args_training = parser_training.parse_args()


    position_encoder = PositionalEncoder(args_training.number_frequencies_postitional,
                                         args_training.use_identity_positional)
    direction_encoder = PositionalEncoder(args_training.number_frequencies_directional,
                                          args_training.use_identity_directional)
    model_coarse = RenderRayNet(args_training.netdepth, args_training.netwidth, position_encoder.output_dim * 3,
                                direction_encoder.output_dim * 3, skips=args_training.skips)
    model_fine = RenderRayNet(args_training.netdepth_fine, args_training.netwidth_fine,
                              position_encoder.output_dim * 3,
                              direction_encoder.output_dim * 3, skips=args_training.skips_fine)
    model_coarse.load_state_dict(
        torch.load(os.path.join(args_inference.run_dir, "model_coarse.pt"), map_location=torch.device('cpu')))
    model_coarse.eval()
    model_fine.load_state_dict(
        torch.load(os.path.join(args_inference.run_dir, "model_fine.pt"), map_location=torch.device('cpu')))
    model_fine.eval()

    transform = transforms.Compose(
        [NormalizeRGB(), CoarseSampling(args_training.near, args_training.far, args_training.number_coarse_samples), ToTensor()])

    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    model_coarse.to(device)
    model_fine.to(device)
    rgb_images = []
    
    if args_inference.model_type == "smpl_nerf":
        dataset = SmplNerfDataset(args_inference.ground_truth_dir, 
                              os.path.join(args_inference.ground_truth_dir, 
                           'transforms.json'), transform)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=args_training.batchsize, shuffle=False, num_workers=0)
        human_pose_encoder = PositionalEncoder(args_training.number_frequencies_pose, args_training.use_identity_pose)
        positions_dim = position_encoder.output_dim if args_training.human_pose_encoding else 1
        human_pose_dim = human_pose_encoder.output_dim if args_training.human_pose_encoding else 1
        model_warp_field = WarpFieldNet(args_training.netdepth_warp, args_training.netwidth_warp, positions_dim * 3,
                                        human_pose_dim * 2)
        model_warp_field.load_state_dict(torch.load(os.path.join(args_inference.run_dir, "model_warp_field.pt")))
        model_warp_field.eval()
        pipeline = SmplNerfPipeline(model_coarse, model_fine, model_warp_field, 
                                    args_training, position_encoder, direction_encoder, human_pose_encoder)
    elif args_inference.model_type == "append_to_nerf":
        human_pose_encoder = PositionalEncoder(args_training.number_frequencies_pose, args_training.use_identity_pose)
        human_pose_dim = human_pose_encoder.output_dim if args_training.human_pose_encoding else 1
        model_coarse = RenderRayNet(args_training.netdepth, args_training.netwidth, position_encoder.output_dim * 3,
                                    direction_encoder.output_dim * 3, human_pose_dim * 2,
                                    skips=args_training.skips)
        model_fine = RenderRayNet(args_training.netdepth_fine, args_training.netwidth_fine,
                                  position_encoder.output_dim * 3,
                                  direction_encoder.output_dim * 3, human_pose_dim * 2,
                                  skips=args_training.skips_fine)
        model_coarse.load_state_dict(
            torch.load(os.path.join(args_inference.run_dir, "model_coarse.pt"), map_location=torch.device('cpu')))
        model_coarse.eval()
        model_fine.load_state_dict(
            torch.load(os.path.join(args_inference.run_dir, "model_fine.pt"), map_location=torch.device('cpu')))
        model_fine.eval()
        dataset = SmplNerfDataset(args_inference.ground_truth_dir,
                                  os.path.join(args_inference.ground_truth_dir,
                                               'transforms.json'), transform)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=args_training.batchsize, shuffle=False,
                                                  num_workers=0)
        human_pose_encoder = PositionalEncoder(args_training.number_frequencies_pose, args_training.use_identity_pose)
        pipeline = AppendToNerfPipeline(model_coarse, model_fine, args_training, position_encoder, direction_encoder, human_pose_encoder)
    elif args_inference.model_type == "smpl":
        dataset = SmplDataset(args_inference.ground_truth_dir, 
                              os.path.join(args_inference.ground_truth_dir, 
                           'transforms.json'), args_training, 
                            transform=NormalizeRGB())
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=args_training.batchsize, shuffle=False, num_workers=0)
        pipeline = SmplPipeline(model_coarse, args_training, position_encoder, direction_encoder)
    elif args_inference.model_type == 'nerf':
        dataset = RaysFromImagesDataset(args_inference.ground_truth_dir, 
                              os.path.join(args_inference.ground_truth_dir, 
                           'transforms.json'), transform)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=args_training.batchsize, shuffle=False, num_workers=0)
        pipeline = NerfPipeline(model_coarse, args_training, position_encoder, direction_encoder)
    camera_transforms = dataset.image_transform_map
    for i, data in enumerate(data_loader):
            for j, element in enumerate(data):
                data[j] = element.to(device)
            rgb_truth = data[-1]
            out = pipeline(data)
            rgb_fine = out[1]
            rgb_images.append(rgb_fine.detach().cpu().numpy())
    rgb_images = np.concatenate(rgb_images, 0).reshape((len(camera_transforms), dataset.h, dataset.w, 3))
    rgb_images = np.clip(rgb_images, 0, 1) * 255
    
    rgb_images = rgb_images.astype(np.uint8)
    save_rerenders(rgb_images, args_inference.run_dir, args_inference.save_dir)
    return rgb_images


def save_rerenders(rgb_images, run_file, output_dir='renders'):
    basename = os.path.basename(run_file)
    output_dir = os.path.join(output_dir, os.path.splitext(basename)[0])
    if not os.path.exists(output_dir):  # create directory if it does not already exist
        os.makedirs(output_dir)
    for i, image in enumerate(rgb_images):
        cv2.imwrite(os.path.join(output_dir, 'img_{:03d}.png'.format(i)), image)
    imageio.mimwrite(os.path.join(output_dir, 'animated.mp4'), rgb_images,
                     fps=30, quality=8)
def config_parser_inference():
    """
    Configuration parser for inference.

    """
    parser = configargparse.ArgumentParser()
    # General
    parser.add_argument('--save_dir', default="renders", help='save directory for inference output (appended to run_dir')
    parser.add_argument('--run_dir', default="runs/Jun20_07-52-59_korhal", help='path to load model')
    parser.add_argument('--ground_truth_dir', default="data/render_append_to_nerf/train",
                        help='path to load ground truth, created with create_dataset.py')
    parser.add_argument('--model_type', default="append_to_nerf", type=str,
                        help='choose dataset type for model [smpl_nerf, nerf, pix2pix, smpl, append_to_nerf]')
    return parser

if __name__ == '__main__':
    rgb_images = inference()
    print(rgb_images.shape)
    plt.imshow(rgb_images[0])
    plt.show()