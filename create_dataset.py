# -*- coding: utf-8 -*-
import numpy as np
import os
from render import get_smpl_mesh, render_scene, save_render, get_human_poses, get_warp
from utils import disjoint_indices
from camera import get_sphere_poses, get_pose_matrix, get_circle_poses, get_circle_on_sphere_poses
import json
from skimage.color import gray2rgb
import configargparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from util.smpl_sequence_loading import load_pose_sequence

np.random.seed(0)


def config_parser():
    """
    Configuration parser for training.

    """
    parser = configargparse.ArgumentParser()
    # General
    parser.add_argument('--save_dir', default="data", help='save directory for dataset')
    parser.add_argument('--dataset_type', default="nerf", type=str,
                        help='choose dataset type for model [smpl_nerf, nerf, pix2pix, smpl]')
    parser.add_argument('--train_val_ratio', default=0.8, type=float, help='train validation ratio')
    # Camera
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
    parser.add_argument('--train_index', default=[], help='Needed to retain the original dataset order', action="append")
    parser.add_argument('--val_index', default=[], help='Needed to retain the original dataset order', action="append")
    parser.add_argument('--smpl_sequence_file', default=None,
                        type=str, help='Path to load sequence of smpl parameters')
    parser.add_argument('--sequence_start', default=0,
                        type=int, help='Sequence start time point')
    parser.add_argument('--sequence_skip', default=3,
                        type=int, help='Sequence skips [::skip]')
    parser.add_argument('--texture', default=1,
                        type=int, help='texture of person 1 to 3')
    parser.add_argument('--sequence_end', default=-1,
                        type=int, help='Sequence end time point')
    parser.add_argument('--frames_per_view', default=1,
                        type=int, help='Frames per view for circle_on_sphere')
    


    return parser


def save_split(save_dir, camera_transforms, indices, split,
               height, width, camera_angle_x, far, dataset_type, human_poses=None, texture=1):
    mesh_canonical, betas, expression = get_smpl_mesh(return_betas_exps=True)
    if dataset_type not in ["nerf", "pix2pix", "smpl_nerf", "smpl"]:
        raise Exception("This dataset type is unknown")
    directory = os.path.join(save_dir, split)
    if not os.path.exists(directory):
        os.makedirs(directory)
    if texture == 1:
        smpl_path = 'SMPLs/smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl'
        texture_path ='textures/female1.jpg'
    elif texture == 2:
        smpl_path = 'SMPLs/smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl'
        texture_path = 'textures/female2.jpg'
    elif texture == 3:
        smpl_path = 'SMPLs/smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl'
        texture_path = 'textures/female3.jpg'
    camera_transforms = camera_transforms[indices]
    image_names = ["img_{:03d}.png".format(index) for index in indices]
    depth_names = ["depth_{:03d}.npy".format(index) for index in indices]
    warp_names = ["warp_{:03d}.npy".format(index) for index in indices]
    print("Length of {} set: {}".format(split, len(image_names)))
    image_transform_map = {image_name: camera_transform.tolist()
                           for (image_name, camera_transform) in zip(image_names, camera_transforms)}
    if dataset_type == "smpl_nerf" or dataset_type == "smpl":
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
            mesh_goal = get_smpl_mesh(body_pose=human_poses[i],smpl_file_name=smpl_path, texture_file_name=texture_path)
            img = render_scene(mesh_goal, camera_pose, get_pose_matrix(), camera_pose,
                               height, width, camera_angle_x)
        elif dataset_type == "smpl":
            mesh_goal = get_smpl_mesh(body_pose=human_poses[i],smpl_file_name=smpl_path, texture_file_name=texture_path)
            trimesh_goal = get_smpl_mesh(body_pose=human_poses[i], return_pyrender=False,smpl_file_name=smpl_path, texture_file_name=texture_path)
            trimesh_canonical = get_smpl_mesh(return_pyrender=False,smpl_file_name=smpl_path, texture_file_name=texture_path)
            img, depth = render_scene(mesh_goal, camera_pose, get_pose_matrix(), camera_pose,
                                      height, width, camera_angle_x, return_depth=True)
            warp, depth1 = get_warp(trimesh_canonical, trimesh_goal, np.array(camera_pose), height, width, camera_angle_x)
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
    human_poses = None
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
    if args.smpl_sequence_file != None:
        human_poses, _ = load_pose_sequence(args.smpl_sequence_file, device="cpu")
        human_poses = human_poses[args.sequence_start:args.sequence_end:args.sequence_skip] #human_poses = human_poses[160::5]
        args.human_number_steps = len(human_poses)
        print(human_poses.shape)
        print(args.human_number_steps)
        if args.multi_human_pose:
            dataset_size = dataset_size * args.human_number_steps
        else:
            dataset_size = len(human_poses)
    elif args.dataset_type == "smpl_nerf" or args.dataset_type == "smpl":
        if args.multi_human_pose:
            dataset_size = dataset_size * args.human_number_steps
    print("Dataset size: ", dataset_size)
    far = args.camera_radius * 2  # For depth normalization

    if args.camera_path == "sphere":
        camera_transforms, camera_angles = get_sphere_poses(args.start_angle, args.end_angle, args.number_steps,
                                                            args.camera_radius)
    elif args.camera_path == "circle":
        camera_transforms, camera_angles = get_circle_poses(args.start_angle, args.end_angle, args.number_steps,
                                                            args.camera_radius)
    elif args.camera_path == "circle_on_sphere":
        camera_transforms, camera_angles = get_circle_on_sphere_poses(dataset_size, 30,
                                                                      args.camera_radius)
        if args.smpl_sequence_file is not None:
            circle_on_sphere_steps = int(dataset_size / args.frames_per_view)
            camera_transforms, camera_angles = get_circle_on_sphere_poses(circle_on_sphere_steps, 30,
                                                                      args.camera_radius)
            
        camera_number_steps = len(camera_transforms)
            
    if (args.dataset_type == "smpl_nerf" or args.dataset_type == "smpl") and args.smpl_sequence_file is None:
        if args.multi_human_pose:
            human_poses = get_human_poses(args.joints, args.human_start_angle, args.human_end_angle,
                                          args.human_number_steps)
            human_poses = human_poses.repeat(camera_number_steps, 1, 1)
            camera_transforms = np.repeat(camera_transforms, args.human_number_steps, axis=0)
        else:
            human_poses = get_human_poses(args.joints, args.human_start_angle, args.human_end_angle,
                                          dataset_size)
    elif args.smpl_sequence_file is not None:
        if args.multi_human_pose:
            human_poses = human_poses.repeat(camera_number_steps, 1, 1)
            camera_transforms = np.repeat(camera_transforms, args.human_number_steps, axis=0)
            print("Human pose: ", human_poses.shape)
            print("Camera trafo: ", camera_transforms.shape)
        else:
            print("Original camera trafo", len(camera_transforms))
            print("Camera steps: ", camera_number_steps)
            print("Human steps: ", args.human_number_steps)
            print(camera_transforms.shape)
            print("Factor: ", int(np.ceil(args.human_number_steps/args.number_steps)))
            if args.frames_per_view == 1:
                camera_transforms = np.concatenate([camera_transforms]*int(np.ceil(args.human_number_steps/camera_number_steps)), axis=0)
            else:
                camera_transforms = np.repeat(camera_transforms, int(np.ceil(args.human_number_steps/camera_number_steps)), axis=0)
            print(camera_transforms.shape)
    train_indices, val_indices = disjoint_indices(dataset_size, args.train_val_ratio)
    train_indices, val_indices = sorted(train_indices), sorted(val_indices)
    save_split(args.save_dir, camera_transforms, train_indices, "train",
               args.resolution, args.resolution, camera_angle_x, far,
               args.dataset_type, human_poses, args.texture)
    save_split(args.save_dir, camera_transforms, val_indices, "val",
               args.resolution, args.resolution, camera_angle_x, far,
               args.dataset_type, human_poses, args.texture)

    args.train_index = train_indices
    args.val_index = val_indices

    parser.write_config_file(args, [os.path.join(args.save_dir, 'create_dataset_config.txt')])


if __name__ == "__main__":
    create_dataset()
