import os
import imageio
import numpy as np
import matplotlib.pyplot as plt
import torch
import glob
from typing import Tuple

from util.scores import print_scores


def load_images(data_root: str, model_type: str="pix2pix"):
    renders = []
    truths = []
    if model_type == "pix2pix":
        print("Load truth images and pix2pix renders from: ", data_root)
        renders_names = sorted(glob.glob(os.path.join(data_root,
                                                      "img_*_fake_B.png")))
        truth_names = sorted(glob.glob(os.path.join(data_root,
                                                    "img_*_real_B.png")))
        for (renders_name, truth_name) in zip(renders_names, truth_names):
            truth = imageio.imread(truth_name)
            render = imageio.imread(renders_name)
            renders.append(render/255.)
            truths.append(truth/255.)
        print("Loaded {} truths and {} pix2pix renders".format(len(truths), len(renders)))
        return truths, renders
    elif model_type == "append_smpl_params":
        print("Load append_smpl_params_nerf rednders from: ", data_root)
        renders_names = sorted(glob.glob(os.path.join(data_root,
                                                    "img_*.png")))
        for (renders_name) in renders_names:
            render = imageio.imread(renders_name)
            renders.append(render/255.)
        print("Loaded {} append_smpl_params_nerf renders".format(len(renders)))
        return renders


def plot_images_side_by_side(imgs, img_captions=['GT', 'SMPLNeRF', 'Pix2Pix']):
    f, axarr = plt.subplots(1, len(imgs))
    assert(len(imgs) == len(img_captions))
    for idx, img in enumerate(imgs):
        axarr[idx].set_xticks([])
        axarr[idx].set_yticks([])
        axarr[idx].imshow(img)
        axarr[idx].set_title(img_captions[idx])
    plt.subplots_adjust(wspace=0, hspace=0)
    f.set_dpi(400)
    f.canvas.draw() # draw the canvas, cache the renderer
    image = np.frombuffer(f.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(f.canvas.get_width_height()[::-1] + (3,))
    plt.clf()
    return image


sequence_name = "sequence_2"
model_name = "Aug24_10-50-14_korhal"
truths, renders_pix2pix = load_images("baseline/pytorch-CycleGAN-and-pix2pix/results/{}/test_latest/images".format(sequence_name), "pix2pix")
renders = load_images("renders/{}".format(sequence_name), "append_smpl_params")
print(len(renders), len(truths), len(renders_pix2pix))
#print_scores(torch.Tensor(truths).permute(0, 3, 1, 2), torch.Tensor(renders).permute(0, 3, 1, 2))
truths = truths.detach().numpy()
renders = renders.detach().numpy()
renders_pix2pix = renders_pix2pix.detach().numpy()
print(truths.shape)
imageio.mimsave('results/renders.gif', [plot_images_side_by_side([truths[i],
                renders[i], renders_pix2pix[i]]) for i in range(len(renders))], 
                fps=10)
print(truths.shape)
print(renders.shape)
