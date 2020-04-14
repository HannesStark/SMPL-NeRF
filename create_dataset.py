# -*- coding: utf-8 -*-
import numpy as np
import os
from render import get_smpl_mesh, render_scene, save_render
from utils import disjoint_indices
from camera import get_sphere_poses, get_pose_matrix
import pickle

np.random.seed(0)


def save_split(save_dir, camera_poses, indices, split,
               height, width, yfov, mesh):
    directory = os.path.join(save_dir, split)
    if not os.path.exists(directory):
        os.makedirs(directory)
    camera_poses = camera_poses[indices]
    
    image_names = ["img_{:03d}.png".format(index) for index in indices]
    print("Length of {} set: {}".format(split, len(image_names)))
    image_transform_map = {image_name: camera_pose
                           for (image_name, camera_pose) in zip(image_names, camera_poses)}
    camera_angle_x = yfov / 2 # camera angle x is half field of view  
    dict = {'camera_angle_x': camera_angle_x, 
            'image_transform_map': image_transform_map} 
    
    for image_name, camera_pose in image_transform_map.items():
        rgb = render_scene(mesh, camera_pose, get_pose_matrix(), camera_pose, 
                           height, width, yfov)
        save_render(rgb, os.path.join(directory, image_name))
    print("Saved {} images under: {}".format(split, directory))
    pkl_file_name = os.path.join(directory, 'transforms.pkl')
    with open(pkl_file_name, 'wb') as handle:
        pickle.dump(dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Saved {} images to transforms map under: {} \n".format(split, pkl_file_name))


def create_dataset():
    height, width, yfov = 512, 512, np.pi / 3
    camera_radius = 2.4
    start_angle, end_angle, number_steps = -90, 90, 5
    
    train_val_ratio = 0.8
    dataset_size = number_steps ** 2
    save_dir = "data/"
    
    smpl_file_name = "SMPLs/smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl"
    texture_file_name = 'textures/texture.jpg'
    uv_map_file_name = 'textures/smpl_uv_map.npy'
    mesh = get_smpl_mesh(smpl_file_name, texture_file_name, uv_map_file_name)
    
    camera_poses = get_sphere_poses(start_angle, end_angle, number_steps, 
                                    camera_radius)
    
    train_indices, val_indices = disjoint_indices(dataset_size, train_val_ratio)
    train_indices, val_indices = sorted(train_indices), sorted(val_indices)
        
    save_split(save_dir, camera_poses, train_indices, "train", 
               height, width, yfov, mesh)
    save_split(save_dir, camera_poses, val_indices, "val",
               height, width, yfov, mesh)
    
if __name__ == "__main__":
    create_dataset()