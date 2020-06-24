# -*- coding: utf-8 -*-
import random

import matplotlib.pyplot as plt
import numpy as np
import os
import configargparse
from vedo import show, Spheres
import pyrender
import trimesh
from tqdm import tqdm

np.random.seed(0)


def config_parser():
    parser = configargparse.ArgumentParser()

    parser.add_argument('--run_dir', default="newest",
                        help='directory created by tensorboard with a densities folder inside. If it is "newest" it will choose the folder that was last modified')
    parser.add_argument('--epoch', default=0, type=int,
                        help='if 0 it will choose the newest epoch')
    parser.add_argument('--number_images', default=2, type=int,
                        help='images that will be visualized')
    parser.add_argument('--number_of_points_to_visualize', default=20000, type=int,
                        help='only visualize this many points. Chooses the highest points of the provided points ')
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

    if args.epoch == 0:
        try:
            filenames = os.listdir(os.path.join(run_dir, 'pyrender_data'))
        except:
            raise ValueError("There seems to be no pyrender data generated for the specified run since the path ",
                             os.path.join(run_dir, 'pyrender_data'), '  was not found')

        if len(filenames) == 0:
            raise ValueError('No epoch in the pyrender_data folder')
        epoch = len(filenames)
    else:
        epoch = args.epoch

    print(run_dir)
    densities_samples_warps = np.load(
        os.path.join(run_dir, 'pyrender_data', "densities_samples_warps" + str(epoch) + '.npz'))
    densities, samples = densities_samples_warps['densities'], densities_samples_warps['samples']
    try:
        warps = densities_samples_warps['warps']
    except:
        warps = None

    if len(densities) < args.number_images:
        number_renders = len(densities)
    else:
        number_renders = args.number_images

    ats = []
    images = []
    for image_index in range(number_renders):
        max_density = np.max(densities[image_index])
        if max_density == 0:
            print('Every density for image ', image_index,
                  ' is 0 so your images are probably white and this visualization has spheres of radius 0')
        normalized_densities = densities[image_index] / max_density

        densities_distribution = normalized_densities / normalized_densities.sum()

        sampled_indices = np.random.choice(np.arange(len(normalized_densities)),
                                           args.number_of_points_to_visualize, p=densities_distribution)
        radii = []
        points = []
        for index in sampled_indices:
            radii.append(normalized_densities[index])
            points.append(samples[image_index][index])
        radii = np.array(radii) * 0.1

        ats.append(image_index)
        images.append(Spheres(points, r=radii, c="lb", res=8))

    show(images, at=ats, axes=2)


if __name__ == "__main__":
    visualize_log_data()
