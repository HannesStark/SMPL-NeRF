import pickle

from torch.utils.data import Subset

import numpy as np
import torch
from torchvision.transforms import transforms

from datasets.rays_from_images_dataset import RaysFromImagesDataset
from datasets.transforms import NormalizeRGB, CoarseSampling, ToTensor

#dataset = RaysFromImagesDataset('images', 'testposes.pkl',
#                                transform=transforms.Compose([NormalizeRGB(), CoarseSampling(2, 6, 64), ToTensor()]))
#val_data = Subset(dataset, np.random.choice(np.arange(len(dataset)), 100))
#train_data = Subset(dataset, np.random.choice(np.arange(len(dataset)), 1024))
#
#train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True, num_workers=0)
#val_loader = torch.utils.data.DataLoader(val_data, batch_size=50, shuffle=False, num_workers=0)
#print(dataset.focal)
#print(dataset.w)

rgb = (np.array([]) / 255.).astype(np.float32)
print(rgb)