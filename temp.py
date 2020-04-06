from torchvision.transforms import transforms

from datasets.rays_dataset import RaysDataset
from datasets.transforms import Normalize, RaySampling
import ctypes
ctypes.cdll.LoadLibrary('caffe2_nvrtc.dll')

import torch

from models.render_ray_net import RenderRayNet
from utils import positional_encoding
from torchsummary import summary

position = torch.Tensor([1,2,3])
direction = torch.Tensor([1,0,2])

position = torch.stack([position, position, position])
direction = torch.stack([direction, direction, direction])

encoded_positions =positional_encoding(position, 10, True)
encoded_directions =positional_encoding(direction, 4, False)
print(encoded_positions.shape)

input = torch.cat([positional_encoding(position, 10, True), positional_encoding(direction, 4, False)], -1)


model = RenderRayNet(n_layers=8, width=256,positions_dim=encoded_positions.shape[0], directions_dim=encoded_directions.shape[0])
outpu = model(input)
print(outpu)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

print(input.shape[0])
summary(model, input_size=tuple([input.shape[1]]))
print(model)


