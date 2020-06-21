# -*- coding: utf-8 -*-
import numpy as np
import os
import configargparse
import pyrender
import trimesh

np.random.seed(0)


def config_parser():
    parser = configargparse.ArgumentParser()

    parser.add_argument('--run_dir', default="newest",
                        help='directory created by tensorboard with a densities folder inside. If it is "newest" it will choose the folder that was last modified')
    parser.add_argument('--epoch', default="newest", type=str,
                        help='by default it will choose the newest epoch')
    return parser


def visualize_log_data():
    parser = config_parser()
    args = parser.parse_args()
    if args.run_dir == "newest":
        run_folders = os.listdir('runs')
        if len(run_folders) == 0:
            raise ValueError('There is no run in the runs directory')
        newest = 0
        run_dir = ""
        for run_folder in run_folders:
            timestamp = os.path.getmtime(os.path.join('runs', run_folder))
            if timestamp > newest:
                newest = timestamp
                run_dir = os.path.join('runs', run_folder)
    else:
        run_dir = args.run_dir

    if args.epoch == "newest":
        filenames = os.listdir(os.path.join(run_dir, 'pyrender_data'))

        if len(filenames) == 0:
            raise ValueError('No epoch in the pyrender_data folder')
        epoch = len(filenames)
    else:
        epoch = args.epoch

    print(run_dir)
    densities_samples_warps = np.load(
        os.path.join(run_dir, 'pyrender_data', "densities_samples_warps" + str(epoch) + '.npz'))
    densities, samples, warps = densities_samples_warps['densities'], densities_samples_warps['samples'], \
                                densities_samples_warps['warps']

    for image_index in range(len(densities)):
        radii = densities[image_index] / np.max(densities[image_index])
        radii = radii * 0.01
        print(radii.shape)
        print(samples[image_index].shape)

        sm = trimesh.creation.uv_sphere(radius=radii)
        sm.visual.vertex_colors = [1.0, 0.0, 0.0]
        tfs = np.tile(np.eye(4), (len(samples[image_index]), 1, 1))
        tfs[:, :3, 3] = samples[image_index]
        m = pyrender.Mesh.from_trimesh(sm, poses=tfs)

        scene = pyrender.Scene()
        scene.add(m)

        pyrender.Viewer(scene, use_raymond_lighting=True)


if __name__ == "__main__":
    visualize_log_data()
