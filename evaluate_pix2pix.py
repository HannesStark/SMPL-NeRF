import os
import imageio
import numpy as np
import matplotlib.pyplot as plt
import torch
import glob

from util.scores import print_scores


def load_images(data_root):
    renders_names = sorted(glob.glob(os.path.join(data_root,"img_*_fake_B.png")))[::3]
    truth_names = sorted(glob.glob(os.path.join(data_root,"img_*_real_B.png")))[::3]
    renders = []
    truths = []
    for (renders_name, truth_name) in zip(renders_names, truth_names):
        truth = imageio.imread(truth_name)
        render = imageio.imread(renders_name)
        renders.append(render/255.)
        truths.append(truth/255.)
    truths = torch.Tensor(truths)
    renders = torch.Tensor(renders)
    return truths, renders

truths, renders = load_images("renders/sequence_walk_turn_left_circle_on_sphere_256_pix2pix_texture_1/images")
print_scores(truths.permute(0, 3, 1, 2), renders.permute(0, 3, 1, 2))
print(truths.shape)
print(renders.shape)