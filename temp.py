from torchsearchsorted.cpu import searchsorted_cpu_wrapper

import torch

b = torch.ones((10,4,3))

b.view(-1,b.shape[-1])

print(b.shape)