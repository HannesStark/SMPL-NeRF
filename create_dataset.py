# -*- coding: utf-8 -*-
import numpy as np
import os
from render import get_smpl_mesh, render_scene, save_render, save_warp, get_human_poses, get_warp
from utils import disjoint_indices
from camera import get_sphere_poses, get_pose_matrix, get_circle_poses, get_circle_on_sphere_poses
import json
from skimage.color import gray2rgb
import configargparse
from tqdm import tqdm

np.random.seed(0)

def config_parser():
    """
    Configuration parser for training.

    """
    parser = configargparse.ArgumentParser()
    # General
    parser.add_argument('--save_dir', default="data", help='save directory for dataset')
    parser.add_argument('--dataset_type', default="nerf", type=str, help='choose dataset type for model [smpl_nerf, nerf, pix2pix]')
    parser.add_argument('--train_val_ratio', default=0.8, type=float, help='train validation ratio')
    # Camera
    parser.add_argument('--resolution', default=128, type=int, help='height and width of renders')
    parser.add_argument('--camera_radius', default=2.4, type=float, help='radius of sphere on which camera moves')
    parser.add_argument('--camera_path', default="sphere", help='Geometric object along which the camera is moved [sphere, circle, circle_on_sphere]')
    parser.add_argument('--start_angle', default=-90, type=int, help='Start angle for phi and theta on sphere')
    parser.add_argument('--end_angle', default=90, type=int, help='End angle for phi and theta on sphere')
    parser.add_argument('--number_steps', default=10, type=int, help='Number of angles inbetween start and end angle')
    # SMPL
    parser.add_argument('--joints', action="append",default=[41, 38], help='List of joints to vary')
    parser.add_argument('--human_start_angle', default=-90, type=int, help='Start angle for human joints')
    parser.add_argument('--human_end_angle', default=90, type=int, help='End angle for human joints')
    parser.add_argument('--human_number_steps', default=10, type=int, help='Number of angles inbetween start and end angle for human joints')

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
            warp = get_warp(trimesh_canonical,trimesh_goal, np.array(camera_pose), height, width, camera_angle_x)
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
        dataset_size = dataset_size * args.human_number_steps
    print("Dataset size: ",dataset_size)
    far = args.camera_radius * 2 # For depth normalization
    
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
        human_poses = get_human_poses(args.joints, args.human_start_angle, args.human_end_angle,
                                      args.human_number_steps)
        human_poses = human_poses.repeat(camera_number_steps, 1, 1)
        camera_transforms = np.repeat(camera_transforms, args.human_number_steps, axis=0)
    train_indices, val_indices = disjoint_indices(dataset_size, args.train_val_ratio)
    train_indices, val_indices = sorted(train_indices), sorted(val_indices)
    save_split(args.save_dir, camera_transforms, train_indices, "train",
               args.resolution, args.resolution, camera_angle_x, far,
               args.dataset_type, human_poses)
    save_split(args.save_dir, camera_transforms, val_indices, "val",
               args.resolution, args.resolution, camera_angle_x, far,
               args.dataset_type, human_poses)
    
if __name__ == "__main__":
    create_dataset()
