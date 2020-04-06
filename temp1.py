import tensorflow as tf
import torch
dists = torch.Tensor([[1,2,3],[1,2,3]])

res = torch.Tensor([1e10]).expand(dists[..., :1].shape)

print(res)