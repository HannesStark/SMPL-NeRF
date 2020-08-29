import os
import imageio
import numpy as np
import matplotlib.pyplot as plt
import torch
import glob
from typing import Tuple

from util.scores import print_scores


def load_images(data_root: str, model_type: str="pix2pix"
                ) -> Tuple(torch.Tensor, torch.Tensor):
    if model_type == "pix2pix":
        renders_names = sorted(glob.glob(os.path.join(data_root,
                                                      "img_*_fake_B.png")))
        truth_names = sorted(glob.glob(os.path.join(data_root,
                                                    "img_*_real_B.png")))
    elif model_type == "append_smpl_params":
        renders_names = sorted(glob)
    renders = []
    truths = []
    for (renders_name, truth_name) in zip(renders_names, truth_names):
        truth = imageio.imread(truth_name)
        render = imageio.imread(renders_name)
        renders.append(render/255.)
        truths.append(truth/255.)
    return torch.Tensor(truths), torch.Tensor(renders)


def plot_images_side_by_side(imgs, img_captions=['GT', 'SMPLNeRF', 'Pix2Pix']):
    f, axarr = plt.subplots(1, len(imgs))
    for idx, img in enumerate(range(imgs)):
        axarr[0, idx].set_xticks([])
        axarr[0, idx].set_yticks([])
        axarr[0, idx].imshow(img)
        axarr[0, idx].set_title(img_captions[idx])
    f.canvas.draw()       # draw the canvas, cache the renderer
    image = np.frombuffer(f.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(f.canvas.get_width_height()[::-1] + (3,))
    return image


sequence_name = "sequence_1"
model_name = "Aug24_10-50-14_korhal"
truths, renders = load_images("""baseline/pytorch-CycleGAN-and-pix2pix/
                              results/{}/test_latest/
                              images""".format(sequence_name), "pix2pix")
print_scores(truths.permute(0, 3, 1, 2), renders.permute(0, 3, 1, 2))
imageio.mimsave('results/renders.gif', [plot_images_side_by_side(truths[i],
                renders[i], renders_pix2pix[i]) for i in range(len(renders))], 
                fps=10)
print(truths.shape)
print(renders.shape)
