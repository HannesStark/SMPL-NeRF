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
from utils import PositionalEncoder

import numpy as np
import os
from render import get_smpl_mesh, render_scene, save_render, get_human_poses, get_warp
from utils import disjoint_indices
from camera import get_sphere_poses, get_pose_matrix, get_circle_poses, get_circle_on_sphere_poses
import json
from skimage.color import gray2rgb
import configargparse
from tqdm import tqdm
import configargparse
from config_parser import config_parser
from models.render_ray_net import RenderRayNet
from models.warp_field_net import WarpFieldNet
from models.smpl_pipeline import SmplPipeline
from datasets.smpl_dataset import SmplDataset
from datasets.transforms import NormalizeRGB


def inference(batch_size=128):
    parser_inference = config_parser_inference()
    args_inference = parser_inference.parse_args()
    parser_training = config_parser()
    config_file_training = os.path.join(args_inference.run_dir, "config.txt")
    parser_training.add_argument('--config2', is_config_file=True, default=config_file_training, help='config file path')
    args_training = parser_training.parse_args()
    camera_transforms, camera_angles = get_circle_on_sphere_poses(args_inference.number_steps, 20,
                                                                      args_inference.camera_radius)
    position_encoder = PositionalEncoder(args_training.number_frequencies_postitional, args_training.use_identity_positional)
    direction_encoder = PositionalEncoder(args_training.number_frequencies_directional, args_training.use_identity_directional)
    model_coarse = RenderRayNet(args_training.netdepth, args_training.netwidth, position_encoder.output_dim * 3,
                                direction_encoder.output_dim * 3, skips=args_training.skips)
    model_fine = RenderRayNet(args_training.netdepth_fine, args_training.netwidth_fine, position_encoder.output_dim * 3,
                              direction_encoder.output_dim * 3, skips=args_training.skips_fine)
    model_coarse.load_state_dict(torch.load(os.path.join(args_inference.run_dir, "model_coarse.pt"), map_location=torch.device('cpu')))
    model_coarse.eval()
    model_fine.load_state_dict(torch.load(os.path.join(args_inference.run_dir, "model_fine.pt"),map_location=torch.device('cpu')))
    model_fine.eval()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    model_coarse.to(device)
    model_fine.to(device)
    if args_inference.camera_path == "sphere":
        dataset_size = args_inference.number_steps ** 2
        camera_number_steps = args_inference.number_steps ** 2
    elif args_inference.camera_path == "circle":
        dataset_size = args_inference.number_steps
        camera_number_steps = args_inference.number_steps
    elif args_inference.camera_path == "circle_on_sphere":
        dataset_size = args_inference.number_steps
        camera_number_steps = args_inference.number_steps
    number_rays = dataset_size*args_inference.resolution**2
    rgb_images = []
    if args_inference.model_type == "smpl_nerf":
        human_pose_encoder = PositionalEncoder(args_training.number_frequencies_pose, args_training.use_identity_pose)
        positions_dim = position_encoder.output_dim if args_training.human_pose_encoding else 1
        human_pose_dim = human_pose_encoder.output_dim if args_training.human_pose_encoding else 1
        model_warp_field = WarpFieldNet(args_training.netdepth_warp, args_training.netwidth_warp, positions_dim * 3,
                                        human_pose_dim * 2)
        model_warp_field.load_state_dict(torch.load(os.path.join(args_inference.run_dir, "model_warp_field.pt")))
        model_warp_field.eval()
    elif args_inference.model_type == "smpl":
        gt_dir = "data_circle_on_sphere/train"
        dataset = SmplDataset(gt_dir, os.path.join(gt_dir, 'transforms.json'), args_training, transform=NormalizeRGB())
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=args_training.batchsize, shuffle=False, num_workers=0)
        pipeline = SmplPipeline(model_coarse, args_training, position_encoder, direction_encoder)
        for i, data in enumerate(data_loader):
                for j, element in enumerate(data):
                    data[j] = element.to(device)
                rgb_truth = data[-1]
                rgb, rgb_fine = pipeline(data)
                rgb_images.append(rgb_fine.detach().cpu().numpy())
        #for batch in range(0, number_rays, args_training.batchsize):
        #    indices = np.arange(0, args_training.batchsize) + batch
        #    if batch + args_training.batchsize > number_rays:
        #        indices = indices[:batch + args_training.batchsize - number_rays]
        rgb_images = np.concatenate(rgb_images, 0).reshape((len(camera_transforms), dataset.h, dataset.w, 3))
        rgb_images = np.clip(rgb_images, 0, 1) * 255

        return rgb_images.astype(np.uint8)
    
    elif args_inference.model_type == 'nerf':
        pass
    elif args_inference.model_type == 'append_to_nerf':
        human_pose_encoder = PositionalEncoder(args_training.number_frequencies_pose, args_training.use_identity_pose)
        human_pose_dim = human_pose_encoder.output_dim if args_training.human_pose_encoding else 1
        model_coarse = RenderRayNet(args_training.netdepth, args_training.netwidth, position_encoder.output_dim * 3,
                                    direction_encoder.output_dim * 3, human_pose_dim * 2,
                                    skips=args_training.skips)
        model_fine = RenderRayNet(args_training.netdepth_fine, args_training.netwidth_fine, position_encoder.output_dim * 3,
                                  direction_encoder.output_dim * 3, human_pose_dim * 2,
                                  skips=args_training.skips_fine)

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
def config_parser_inference():
    """
    Configuration parser for inference.

    """
    parser = configargparse.ArgumentParser()
    # General
    parser.add_argument('--save_dir', default="data", help='save directory for inference output')
    parser.add_argument('--run_dir', default="runs/Jun03_12-17-11_korhal", help='path to load model')
    parser.add_argument('--model_type', default="smpl", type=str,
                        help='choose dataset type for model [smpl_nerf, nerf, pix2pix, smpl]')    # Camera
    parser.add_argument('--resolution', default=128, type=int, help='height and width of renders')
    parser.add_argument('--camera_radius', default=2.4, type=float, help='radius of sphere on which camera moves')
    parser.add_argument('--camera_path', default="sphere",
                        help='Geometric object along which the camera is moved [sphere, circle, circle_on_sphere]')
    parser.add_argument('--start_angle', default=-90, type=int, help='Start angle for phi and theta on sphere')
    parser.add_argument('--end_angle', default=90, type=int, help='End angle for phi and theta on sphere')
    parser.add_argument('--number_steps', default=10, type=int, help='Number of angles inbetween start and end angle')
    # SMPL
    parser.add_argument('--joints', action="append", default=[41, 38], help='List of joints to vary')
    parser.add_argument('--human_start_angle', default=-90, type=int, help='Start angle for human joints')
    parser.add_argument('--human_end_angle', default=90, type=int, help='End angle for human joints')
    parser.add_argument('--human_number_steps', default=10, type=int,
                        help='Number of angles inbetween start and end angle for human joints')
    parser.add_argument("--multi_human_pose", type=int, default=0,
                        help='Multiple human poses per viewpoint')

    return parser


def save_split(save_dir, camera_transforms, indices, split,
               height, width, camera_angle_x, far, dataset_type, human_poses=None):
    mesh_canonical, betas, expression = get_smpl_mesh(return_betas_exps=True)
    if dataset_type not in ["nerf", "pix2pix", "smpl_nerf", "smpl"]:
        raise Exception("This dataset type is unknown")
    directory = os.path.join(save_dir, split)
    if not os.path.exists(directory):
        os.makedirs(directory)
    camera_transforms = camera_transforms[indices]
    image_names = ["img_{:03d}.png".format(index) for index in indices]
    depth_names = ["depth_{:03d}.npy".format(index) for index in indices]
    warp_names = ["warp_{:03d}.npy".format(index) for index in indices]
    print("Length of {} set: {}".format(split, len(image_names)))
    image_transform_map = {image_name: camera_transform.tolist()
                           for (image_name, camera_transform) in zip(image_names, camera_transforms)}
    if dataset_type == "smpl_nerf" or "smpl":
        human_poses = human_poses[indices]
        image_pose_map = {image_name: human_pose[0].numpy().tolist()
                          for (image_name, human_pose) in zip(image_names, human_poses)}
        dict = {'camera_angle_x': camera_angle_x,
                'image_transform_map': image_transform_map,
                'image_pose_map': image_pose_map,
                'betas': betas[0].numpy().tolist(),
                'expression': expression[0].numpy().tolist()}
    elif dataset_type == "nerf" or dataset_type == "pix2pix":
        dict = {'camera_angle_x': camera_angle_x,
                'image_transform_map': image_transform_map}
    for i, (image_name, camera_pose) in tqdm(enumerate(image_transform_map.items())):
        if dataset_type == "nerf":
            img = render_scene(mesh_canonical, camera_pose, get_pose_matrix(), camera_pose,
                               height, width, camera_angle_x)
        elif dataset_type == "pix2pix":
            rgb, depth = render_scene(mesh_canonical, camera_pose, get_pose_matrix(), camera_pose,
                                      height, width, camera_angle_x, return_depth=True)
            depth = (depth / far * 255).astype(np.uint8)
            img = np.concatenate([rgb, gray2rgb(depth)], 1)
        elif dataset_type == "smpl_nerf":
            mesh_goal = get_smpl_mesh(body_pose=human_poses[i])
            img = render_scene(mesh_goal, camera_pose, get_pose_matrix(), camera_pose,
                               height, width, camera_angle_x)
        elif dataset_type == "smpl":
            mesh_goal = get_smpl_mesh(body_pose=human_poses[i])
            trimesh_goal = get_smpl_mesh(body_pose=human_poses[i], return_pyrender=False)
            trimesh_canonical = get_smpl_mesh(return_pyrender=False)
            img, depth = render_scene(mesh_goal, camera_pose, get_pose_matrix(), camera_pose,
                                      height, width, camera_angle_x, return_depth=True)
            warp = get_warp(trimesh_canonical, trimesh_goal, np.array(camera_pose), height, width, camera_angle_x)
            np.save(os.path.join(directory, warp_names[i]), warp)
            np.save(os.path.join(directory, depth_names[i]), depth)
        save_render(img, os.path.join(directory, image_name))

    print("Saved {} images under: {}".format(split, directory))
    json_file_name = os.path.join(directory, 'transforms.json')
    with open(json_file_name, 'w') as fp:
        json.dump(dict, fp)
    print("Saved {} images to transforms map under: {} \n".format(split, json_file_name))


def create_dataset():
    parser = config_parser()
    args = parser.parse_args()
    camera_angle_x = np.pi / 3
    if args.camera_path == "sphere":
        dataset_size = args.number_steps ** 2
        camera_number_steps = args.number_steps ** 2
    elif args.camera_path == "circle":
        dataset_size = args.number_steps
        camera_number_steps = args.number_steps
    elif args.camera_path == "circle_on_sphere":
        dataset_size = args.number_steps
        camera_number_steps = args.number_steps
    else:
        raise Exception("This camera path is unknown")
    if args.dataset_type == "smpl_nerf" or args.dataset_type == "smpl":
        if args.multi_human_pose:
            dataset_size = dataset_size * args.human_number_steps
    print("Dataset size: ", dataset_size)
    far = args.camera_radius * 2  # For depth normalization

    human_poses = None
    if args.camera_path == "sphere":
        camera_transforms, camera_angles = get_sphere_poses(args.start_angle, args.end_angle, args.number_steps,
                                                            args.camera_radius)
    elif args.camera_path == "circle":
        camera_transforms, camera_angles = get_circle_poses(args.start_angle, args.end_angle, args.number_steps,
                                                            args.camera_radius)
    elif args.camera_path == "circle_on_sphere":
        camera_transforms, camera_angles = get_circle_on_sphere_poses(args.number_steps, 20,
                                                                      args.camera_radius)
    if args.dataset_type == "smpl_nerf" or args.dataset_type == "smpl":
        if args.multi_human_pose:
            human_poses = get_human_poses(args.joints, args.human_start_angle, args.human_end_angle,
                                          args.human_number_steps)
            human_poses = human_poses.repeat(camera_number_steps, 1, 1)
            camera_transforms = np.repeat(camera_transforms, args.human_number_steps, axis=0)
        else:
            human_poses = get_human_poses(args.joints, args.human_start_angle, args.human_end_angle,
                                          dataset_size)
    train_indices, val_indices = disjoint_indices(dataset_size, args.train_val_ratio)
    train_indices, val_indices = sorted(train_indices), sorted(val_indices)
    save_split(args.save_dir, camera_transforms, train_indices, "train",
               args.resolution, args.resolution, camera_angle_x, far,
               args.dataset_type, human_poses)
    save_split(args.save_dir, camera_transforms, val_indices, "val",
               args.resolution, args.resolution, camera_angle_x, far,
               args.dataset_type, human_poses)

if __name__ == '__main__':
    inference()
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
    """angles = np.linspace(0, np.pi*2, 25)
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
    save_rerenders(rgb_images, run_file)"""
