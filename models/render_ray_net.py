import torch
import torch.nn as nn
import torch.nn.functional as F


class RenderRayNet(nn.Module):

    def __init__(self, n_layers=8, width=256, positions_dim=60, directions_dim=24, additional_input_dim=0, skips=[4], use_directional_input=1):
        super(RenderRayNet, self).__init__()

        self.n_layers = n_layers
        self.width = width
        self.positions_dim = positions_dim
        self.direcions_dim = directions_dim
        self.skips = skips
        self.additional_input_dim = additional_input_dim
        self.use_directional_input = use_directional_input

        self.positions_pose_input = torch.nn.Linear(positions_dim + additional_input_dim, width)
        self.positional_net = nn.ModuleList()
        for i in range(self.n_layers - 1):  # minus one because we create the first layer as self.positions_pose_input
            if i in skips:
                self.positional_net.append(torch.nn.Linear(width + positions_dim + additional_input_dim, width))
            else:
                self.positional_net.append(torch.nn.Linear(width, width))

        self.additional_linear_layer = torch.nn.Linear(width, width)
        self.sigma_out_layer = torch.nn.Linear(width, 1)

        directional_width = width // 2
        if use_directional_input:
            self.directional_input = torch.nn.Linear(width + directions_dim, directional_width)
        else:
            self.directional_input = torch.nn.Linear(width, directional_width)

        self.directional_net = nn.ModuleList()
        # for i in range(self.n_layers // 2 - 1):  # minus one because we create the first layer as self.directional_input
        for i in range(1):
            self.directional_net.append(torch.nn.Linear(directional_width, directional_width))
        self.rgb_out_layer = torch.nn.Linear(directional_width, 3)

    def forward(self, x):
        positions_pose, directions = x[..., :self.positions_dim + self.additional_input_dim], x[..., -self.direcions_dim:]
        o = positions_pose
        o = F.relu(self.positions_pose_input(o))
        for i, positional_layer in enumerate(self.positional_net):
            if i in self.skips:
                o = F.relu(positional_layer(torch.cat([o, positions_pose], -1)))
            else:
                o = F.relu(positional_layer(o))
        o = self.additional_linear_layer(o)
        sigma = self.sigma_out_layer(o)

        if self.use_directional_input:
            o = self.directional_input(torch.cat([o, directions], -1))
        else:
            o = self.directional_input(o)
        for i, directional_layer in enumerate(self.directional_net):
            o = F.relu(directional_layer(o))
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
