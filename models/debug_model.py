import torch
import torch.nn as nn
import torch.nn.functional as F


class DebugModel(nn.Module):

    def __init__(self, n_layers=8, width=256, positions_dim=60, directions_dim=24, skips=[4]):
        super(DebugModel, self).__init__()

        self.n_layers = n_layers
        self.width = width
        self.positions_dim = positions_dim
        self.direcions_dim = directions_dim
        self.skips = skips

        self.positional_input = torch.nn.Linear(positions_dim, width)
        self.linear1 = torch.nn.Linear(width, width)
        self.linear2 = torch.nn.Linear(width, width)
        self.linear3 = torch.nn.Linear(width, width)
        self.linear4 = torch.nn.Linear(width, width)
        self.linear5 = torch.nn.Linear(width, width)
        self.linear6 = torch.nn.Linear(width, width)
        self.linear7 = torch.nn.Linear(width, width)
        self.sigma_out_layer = torch.nn.Linear(width, 1)

        self.rgb_out_layer = torch.nn.Linear(width, 3)

    def forward(self, x):
        positions, directions = x[..., :self.positions_dim], x[..., -self.direcions_dim:]
        o = positions
        o = F.relu(self.positional_input(o))
        # o = F.relu(self.linear1(o))
        # o = F.relu(self.linear2(o))
        # o = F.relu(self.linear3(o))
        # o = F.relu(self.linear4(o))
        # o = F.relu(self.linear5(o))
        o = F.relu(self.linear6(o))
        o = F.relu(self.linear7(o))
        sigma = self.sigma_out_layer(o)
        rgb = self.rgb_out_layer(o)

        return torch.cat([rgb, sigma], -1)

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
