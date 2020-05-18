import torch
import torch.nn as nn
import torch.nn.functional as F


class WarpFieldNet(nn.Module):

    def __init__(self, n_layers=8, width=256, positions_dim=60, pose_dim=24):
        super(WarpFieldNet, self).__init__()

        self.positions_dim = positions_dim
        self.direcions_dim = pose_dim


        self.linear1 = torch.nn.Linear(positions_dim + pose_dim, width)
        self.linear2 = torch.nn.Linear(width, 3)


    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
