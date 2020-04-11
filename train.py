import torch
from torch.utils.data import Subset
from torchvision.transforms import transforms
import matplotlib.pyplot as plt

from datasets.rays_from_images_dataset import RaysFromImagesDataset
from datasets.transforms import CoarseSampling, ToTensor, NormalizeRGB
from models.render_ray_net import RenderRayNet
from solver.solver import Solver
import ctypes
import numpy as np

from utils import PositionalEncoder, save_run

np.random.seed(0)

ctypes.cdll.LoadLibrary('caffe2_nvrtc.dll')

dataset = RaysFromImagesDataset('images', 'testposes.pkl',
                                transform=transforms.Compose([NormalizeRGB(), CoarseSampling(2, 6, 64), ToTensor()]))
val_data = Subset(dataset, np.random.choice(np.arange(len(dataset)), 10))
#train_data = Subset(dataset, np.random.choice(np.arange(len(dataset)), 200))

train_loader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=5, shuffle=False, num_workers=0)

position_encoder = PositionalEncoder(number_frequencies=10, include_identity=True)
direction_encoder = PositionalEncoder(number_frequencies=4, include_identity=False)
model_coarse = RenderRayNet(positions_dim=position_encoder.output_dim * 3,
                            directions_dim=direction_encoder.output_dim * 3)
model_fine = RenderRayNet(positions_dim=position_encoder.output_dim * 3,
                            directions_dim=direction_encoder.output_dim * 3)

solver = Solver(position_encoder, direction_encoder, optim_args={"lr": 1e-3, "weight_decay": 0},
                loss_func=torch.nn.MSELoss(), white_background=True)
solver.train(model_coarse, model_fine, train_loader, val_loader, log_nth=1, num_epochs=100)

save_run('first_experiment', model_coarse, model_fine, dataset, solver)

plt.plot(solver.train_loss_history, label='Train loss')
plt.plot(solver.val_loss_history, label='Val loss')
plt.legend(loc="upper right")

# plt.savefig('/content/drive/My Drive/' + model_name + str(segment_length))
plt.show()
