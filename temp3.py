import cv2
import numpy as np
import smplx
import torch
import trimesh
import torch.nn.functional as F

model = torch.nn.Sequential(torch.nn.Linear(87, 256),
                            torch.nn.ReLU(),
                            torch.nn.Linear(256, 256),
                            torch.nn.ReLU(),
                            torch.nn.Linear(256, 3),
                            torch.nn.Sigmoid())

print(model)
loss_func = torch.nn.MSELoss()
input = torch.rand((64, 87))
img = cv2.imread("data/train/img_000.png")
print(img)
img = img / 255.
img = np.array(img).astype(np.float32)
img = torch.from_numpy(img)

optim = torch.optim.Adam(model.parameters(), lr=5e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

for i in range(10000):
    rgb = model(input)
    rgb = rgb.view(img.shape)

    optim.zero_grad()
    loss = loss_func(rgb, img)
    loss.backward()
    optim.step()
    print("loss ", loss.item())
