import cv2
import numpy as np
import smplx
import torch
import trimesh
import torch.nn.functional as F

from utils import get_gmm_from_smpl, GaussianMixture

b = np.random.random((7, 3))


canonical_mixture = get_gmm_from_smpl(b, "cpu", 0.5)
own_mixture = GaussianMixture(b, 0.5, "cpu")

samples = torch.rand(2, 3)

own_res = own_mixture.pdf(samples)
torch_res = torch.exp(canonical_mixture.log_prob(samples))

print(own_res)
print(torch_res)
