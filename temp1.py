
import torch
import numpy as np

b = torch.Tensor(
    [[1, 2, 3],
     [2, 3, 4]],
)

print(b.view(-1))
print(b)

