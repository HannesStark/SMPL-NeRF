import torch
import torch.nn as nn
import torch.nn.functional as F


class SmplEstimator(nn.Module):

    def __init__(self, human_size=69):
        super(SmplEstimator, self).__init__()

        # input: 128x128x3 image tensor
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        #self.conv6= nn.Conv2d(128, 256, 3, padding=1)
        #self.bn6 = nn.BatchNorm2d(256)

        self.fc1 = nn.Linear(8 * 8 * 128, 500)
        self.dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(500, human_size)

    def forward(self, x):

        # 128
        x = F.relu(self.bn1(self.conv1(x)))  # 128
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 64
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 32
        x = self.pool(F.relu(self.bn4(self.conv4(x))))  # 16
        x = self.pool(F.relu(self.bn5(self.conv5(x))))  # 8
        #x = self.pool(F.relu(self.batchnorm_6(self.conv_23(x))))  # 4
        # flatten image input
        x = x.view(-1, 8 * 8 * 128)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
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