import torch
import numpy as np
np.random.seed(0)
from torchvision.transforms import transforms
import ctypes
ctypes.cdll.LoadLibrary('caffe2_nvrtc.dll')

from datasets.rays_dataset import RaysDataset
from datasets.transforms import CoarseSampling, ToTensor
from models.render_ray_net import RenderRayNet
from utils import positional_encoding, raw2outputs


dataset = RaysDataset('images', 'testposes.pkl', transform=transforms.Compose([CoarseSampling(2, 6, 64), ToTensor()]))

train_loader = torch.utils.data.DataLoader(dataset, batch_size=4096, shuffle=False, num_workers=0)

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"
#if torch.cuda.is_available():
#    torch.set_default_tensor_type('torch.cuda.FloatTensor')

model_coarse = RenderRayNet(positions_dim=63)
model_coarse.to(device)
optim = torch.optim.Adam(model_coarse.parameters())
for i, data in enumerate(train_loader):
    ray_samples, samples_translations, samples_directions, z_vals, rgb_truth = data
    print(z_vals)
    ray_samples = ray_samples.to(device)  # [batchsize, number_coarse_samples, 3]
    samples_translations = samples_translations.to(device)  # [batchsize, number_coarse_samples, 3]
    samples_directions = samples_directions.to(device)  # [batchsize, number_coarse_samples, 3]
    z_vals = z_vals.to(device)  # [batchsize, number_coarse_samples]
    rgb_truth = rgb_truth.to(device)  # [batchsize, 3]

    # get values for coarse network and run them through the coarse network
    samples_encoding = positional_encoding(ray_samples, 10, True)
    samples_directions_norm = samples_directions / torch.norm(samples_directions, dim=-1, keepdim=True)
    directions_encoding = positional_encoding(samples_directions_norm, 4, False)
    # flatten the encodings from [batchsize, number_coarse_samples, encoding_size] to [batchsize * number_coarse_samples, encoding_size] and concatenate
    inputs = torch.cat([samples_encoding.view(-1, samples_encoding.shape[-1]),
                        directions_encoding.view(-1, directions_encoding.shape[-1])], -1)
    print(inputs)
    optim.zero_grad()
    raw_outputs = model_coarse(inputs)  # [batchsize * number_coarse_samples, 4]
    raw_outputs = raw_outputs.view(samples_encoding.shape[0], samples_encoding.shape[1],
                                   raw_outputs.shape[-1])  # [batchsize, number_coarse_samples, 4]
    rgb, weights = raw2outputs(raw_outputs, z_vals, samples_directions, 1,
                               False)

