import os
import cv2
import imageio
import numpy as np
import json
import matplotlib.pyplot as plt
import torch

from torchvision.transforms import transforms

import configargparse
from tqdm import tqdm

from config_parser import config_parser
from models.append_to_nerf_pipeline import AppendToNerfPipeline

from models.render_ray_net import RenderRayNet
from models.warp_field_net import WarpFieldNet
from models.singe_sample_pipeline import SmplPipeline
from models.smpl_nerf_pipeline import SmplNerfPipeline
from models.nerf_pipeline import NerfPipeline
from models.vertex_sphere_pipeline import VertexSpherePipeline

from datasets.smpl_nerf_dataset import SmplNerfDataset
from datasets.rays_from_images_dataset import RaysFromImagesDataset
from datasets.single_sample_dataset import SmplDataset
from datasets.transforms import CoarseSampling, ToTensor, NormalizeRGB

from utils import PositionalEncoder
import create_dataset

from util.scores import print_scores


def inference():
    parser_inference = config_parser_inference()
    args_inference = parser_inference.parse_args()
    parser_training = config_parser()
    config_file_training = os.path.join(args_inference.run_dir, "config.txt")
    parser_training.add_argument('--config2', is_config_file=True,
                     default=config_file_training, help='config file path')
    args_training = parser_training.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    position_encoder = PositionalEncoder(args_training.number_frequencies_postitional,
                                         args_training.use_identity_positional)
    direction_encoder = PositionalEncoder(args_training.number_frequencies_directional,
                                          args_training.use_identity_directional)
    if not args_inference.model_type == "append_to_nerf":
        model_coarse = RenderRayNet(args_training.netdepth, args_training.netwidth, position_encoder.output_dim * 3,
                                    direction_encoder.output_dim * 3, skips=args_training.skips)
        model_fine = RenderRayNet(args_training.netdepth_fine, args_training.netwidth_fine,
                                  position_encoder.output_dim * 3,
                                  direction_encoder.output_dim * 3, skips=args_training.skips_fine)
        model_coarse.load_state_dict(
            torch.load(os.path.join(args_inference.run_dir, "model_coarse.pt"), map_location=torch.device('cpu')))
        model_coarse.eval()
        model_coarse.to(device)
        if os.path.exists(os.path.join(args_inference.run_dir, "model_fine.pt")):
            model_fine.load_state_dict(
            torch.load(os.path.join(args_inference.run_dir, "model_fine.pt"), map_location=torch.device('cpu')))
            model_fine.eval()
            model_fine.to(device)
        else:
            model_fine = None

    transform = transforms.Compose(
        [NormalizeRGB(), CoarseSampling(args_training.near, args_training.far, args_training.number_coarse_samples),
         ToTensor()])

    rgb_images_renders = []
    rgb_images_truth = []

    if args_inference.model_type == "smpl_nerf":
        dataset = SmplNerfDataset(args_inference.ground_truth_dir,
                                  os.path.join(args_inference.ground_truth_dir,
                                               'transforms.json'), transform)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=args_training.batchsize, shuffle=False,
                                                  num_workers=0)
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
        model_coarse.to(device)
        model_fine.to(device)
        dataset = SmplNerfDataset(args_inference.ground_truth_dir,
                                  os.path.join(args_inference.ground_truth_dir,
                                               'transforms.json'), transform)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=args_training.batchsize, shuffle=False,
                                                  num_workers=0)
        human_pose_encoder = PositionalEncoder(args_training.number_frequencies_pose, args_training.use_identity_pose)
        pipeline = AppendToNerfPipeline(model_coarse, model_fine, args_training, position_encoder, direction_encoder,
                                        human_pose_encoder)
    elif args_inference.model_type == "smpl":
        dataset = SmplDataset(args_inference.ground_truth_dir,
                              os.path.join(args_inference.ground_truth_dir,
                                           'transforms.json'), args_training,
                              transform=NormalizeRGB())
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=args_training.batchsize, shuffle=False,
                                                  num_workers=0)
        pipeline = SmplPipeline(model_coarse, args_training, position_encoder, direction_encoder)
    elif args_inference.model_type == 'nerf':
        dataset = RaysFromImagesDataset(args_inference.ground_truth_dir,
                                        os.path.join(args_inference.ground_truth_dir,
                                                     'transforms.json'), transform)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=args_training.batchsize, shuffle=False,
                                                  num_workers=0)
        pipeline = NerfPipeline(model_coarse, model_fine, args_training, position_encoder, direction_encoder)
    camera_transforms = dataset.image_transform_map
    for i, data in enumerate(tqdm(data_loader)):
        for j, element in enumerate(data):
            data[j] = element.to(device)
        rgb_truth = data[-1]
        out = pipeline(data)
        rgb_fine = out[1]
        rgb_images_renders.append(rgb_fine.detach().cpu())
        rgb_images_truth.append(rgb_truth.detach().cpu())
    rgb_images_renders = torch.cat(rgb_images_renders).reshape((len(camera_transforms), dataset.h, dataset.w, 3))
    rgb_images_truth = torch.cat(rgb_images_truth).reshape((len(camera_transforms), dataset.h, dataset.w, 3))
    # calculate scores
    print_scores(rgb_images_renders.permute(0, 3, 1, 2), rgb_images_truth.permute(0, 3, 1, 2))
    # save renders
    rgb_images_renders = np.concatenate(rgb_images_renders.numpy(), 0).reshape((len(camera_transforms), dataset.h, dataset.w, 3))
    rgb_images_renders = np.clip(rgb_images_renders, 0, 1) * 255
    rgb_images_renders = rgb_images_renders.astype(np.uint8)
    rgb_images_renders = rgb_images_renders[..., ::-1]
    save_rerenders(rgb_images_renders, args_inference.run_dir, args_inference.save_dir)
    return rgb_images_renders


def save_rerenders(rgb_images, run_file, output_dir='renders'):
    basename = os.path.basename(run_file)
    output_dir = os.path.join(output_dir, os.path.splitext(basename)[0])
    if not os.path.exists(output_dir):  # create directory if it does not already exist
        os.makedirs(output_dir)
    for i, image in enumerate(rgb_images):
        imageio.imwrite(os.path.join(output_dir, 'img_{:03d}.png'.format(i)), image)
    imageio.mimwrite(os.path.join(output_dir, 'animated.gif'), rgb_images,
                     fps=30)


def config_parser_inference():
    """
    Configuration parser for inference.

    """
    parser = configargparse.ArgumentParser()
    # General
    parser.add_argument('--save_dir', default="renders",
                        help='save directory for inference output (appended to run_dir')
    parser.add_argument('--run_dir', default="runs/Aug18_10-51-09_korhal", help='path to load model')
    parser.add_argument('--ground_truth_dir', default="data/fix_pose_camera_-10_10_10/val",
                        help='path to load ground truth, created with create_dataset.py')
    parser.add_argument('--model_type', default="nerf", type=str,
                        help='choose dataset type for model [smpl_nerf, nerf, pix2pix, smpl, append_to_nerf]')
    return parser


def inference_gif(run_dir, model_type, args, train_data, val_data, position_encoder, direction_encoder, model_coarse, model_fine, model_dependent):
    """
    Create an animated GIF for the whole training distribution (training + validation set)
    in the correct original order

    """

    parser_data = create_dataset.config_parser()
    config_file_data = os.path.join(run_dir, "create_dataset_config.txt")

    parser_data.add_argument('--config_data', is_config_file=True,
                                 default=config_file_data, help='config file path')
    args_create_data = parser_data.parse_args()

    model_coarse.eval()
    model_fine.eval()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    model_coarse.to(device)
    model_fine.to(device)
    rgb_images = []

    dataset = torch.utils.data.ConcatDataset([train_data, val_data])

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batchsize, shuffle=False, num_workers=0)

    if model_type == "smpl_nerf":
        human_pose_encoder, positions_dim, human_pose_dim, model_warp_field = model_dependent
        model_warp_field.eval()
        pipeline = SmplNerfPipeline(model_coarse, model_fine, model_warp_field,
                                    args, position_encoder, direction_encoder, human_pose_encoder)

    elif model_type == "append_to_nerf":
        [human_pose_encoder, human_pose_dim] = model_dependent
        pipeline = AppendToNerfPipeline(model_coarse, model_fine, args, position_encoder, direction_encoder, human_pose_encoder)

    elif model_type == "smpl":
        pipeline = SmplPipeline(model_coarse, model_fine, args, position_encoder, direction_encoder)

    elif model_type == 'nerf':
        print(direction_encoder)
        pipeline = NerfPipeline(model_coarse, model_fine, args, position_encoder, direction_encoder)

    # add inference for new vertex_sphere approach
    elif args.model_type == 'vertex_sphere':
        pipeline = VertexSpherePipeline(model_coarse, model_fine, args, position_encoder, direction_encoder)


    for i, data in enumerate(data_loader):
        for j, element in enumerate(data):
            data[j] = element.to(device)
        rgb_truth = data[-1]
        out = pipeline(data)
        rgb_fine = out[1]
        rgb_images.append(rgb_fine.detach().cpu().numpy())

    # sort according to names in train, val directories
    split_indices = args_create_data.train_index + args_create_data.val_index

    n_images = len(train_data.image_transform_map) + len(val_data.image_transform_map)

    temp = np.concatenate(rgb_images, 0)
    rgb_images = np.vsplit(temp, n_images)

    rgb_images = [image for _, image in sorted(zip(split_indices, rgb_images))]

    rgb_images = np.concatenate(rgb_images, 0).reshape((n_images, train_data.h, train_data.w, 3))
    rgb_images = np.clip(rgb_images, 0, 1) * 255

    rgb_images = rgb_images.astype(np.uint8)

    save_rerenders(rgb_images, run_dir, run_dir + "/animated")
    print("Created Animation of the whole training distribution!")
    return rgb_images

if __name__ == '__main__':
    rgb_images, rgb_images = inference()
    plt.imshow(rgb_images[0])
    plt.show()
