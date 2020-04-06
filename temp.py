from torchvision.transforms import transforms

from datasets.rays_dataset import RaysDataset
from datasets.transforms import Normalize, RaySampling

import torch

from utils import positional_encoding

x =positional_encoding(torch.Tensor(3),10,True)
print(x.shape)