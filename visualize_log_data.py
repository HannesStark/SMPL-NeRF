# -*- coding: utf-8 -*-
import random
import re

import matplotlib.pyplot as plt
import numpy as np
import os
import configargparse
from vedo import show, Spheres, Arrows
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
    parser.add_argument('--mode', default='warp', type=str,
                        help='whether to visualize densities or warps [density, warp]')
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
            filenames = os.listdir(os.path.join(run_dir, 'vedo_data'))
        except:
            raise ValueError("There seems to be no vedo data generated for the specified run since the path ",
                             os.path.join(run_dir, 'vedo_data'), '  was not found')

        if len(filenames) == 0:
            raise ValueError('No epoch in the vedo_data folder')
        epoch = 0

        for filename in filenames:
            file_epoch = int(re.findall(r'\d+', filename)[0])
            if file_epoch > epoch:
                epoch = file_epoch
    else:
        epoch = args.epoch

    ats = []
    images = []

    print('Using run ' + run_dir + ' and epoch ' + str(epoch))

    for image_index in range(args.number_images):
        try:
            densities_samples_warps = np.load(
                os.path.join(run_dir, 'vedo_data',
                             "densities_samples_warps_epoch_" + str(epoch) + '_image_' + str(image_index) + '.npz'))
            densities, densities_samples, warps, warps_samples = densities_samples_warps['densities'], \
                                                                 densities_samples_warps['samples_density'], \
                                                                 densities_samples_warps['warps'], \
                                                                 densities_samples_warps['samples_warp']

            if args.mode == "density":
                max_density = np.max(densities)
                if max_density == 0:
                    print('Every density for image ', image_index,
                          ' is 0 so your images are probably white and this visualization has spheres of radius 0')
                normalized_densities = densities / max_density

                radii = normalized_densities * 0.1
                ats.append(image_index)
                images.append(Spheres(densities_samples, r=radii, c="lb", res=8))
            elif args.mode == "warp":
                ats.append(image_index)
                images.append([Arrows(warps_samples, warps_samples + warps, s=0.3),
                               Spheres(warps_samples, r=0.01, res=8)])

        except FileNotFoundError as err:
            print('Skipping the iteration with image index ', image_index, ' because the file for that image '
                                                                           'was not found: ', err)

    show(images, at=ats, axes=2)


if __name__ == "__main__":
    visualize_log_data()
