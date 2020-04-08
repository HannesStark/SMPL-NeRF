import torch
from torchvision.transforms import transforms
import matplotlib.pyplot as plt

from datasets.rays_dataset import RaysDataset
from datasets.transforms import CoarseSampling, ToTensor
from models.render_ray_net import RenderRayNet
from solver.solver import Solver
import ctypes

ctypes.cdll.LoadLibrary('caffe2_nvrtc.dll')

dataset = RaysDataset('images', 'testposes.pkl', transform=transforms.Compose([CoarseSampling(2, 6, 64), ToTensor()]))

train_loader = torch.utils.data.DataLoader(dataset, batch_size=30, shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(dataset, batch_size=30, shuffle=False, num_workers=0)

model_coarse = RenderRayNet()
model_fine = RenderRayNet()

solver = Solver(optim_args={"lr": 1e-4, "weight_decay": 0}, loss_func=torch.nn.MSELoss())
solver.train(model_coarse, model_fine, train_loader, val_loader, log_nth=10, num_epochs=50)

# model_name = model.__class__.__name__ + ''
# model.save('/content/drive/My Drive/' + model_name + str(segment_length) + '.model')

plt.plot(solver.train_loss_history, label='Train loss')
plt.plot(solver.val_loss_history, label='Val loss')
plt.legend(loc="upper right")

# plt.savefig('/content/drive/My Drive/' + model_name + str(segment_length))
plt.show()
