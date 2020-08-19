import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class DummyImageWiseEstimator(nn.Module):
    '''
            Uses indices instead of images to fetch preset expressions betas and goal_poses that could be for example perturbed
            Is used by the dynamic_pipeline and dynamic_solver just like the real SmplEstimator
    '''

    def __init__(self, canonical_pose1, canonical_pose2, canonical_pose3, arm_angle_l,
                 arm_angle_r, betas, ground_truth_pose=None):
        super(DummyImageWiseEstimator, self).__init__()

        self.canonical_pose1 = torch.nn.Parameter(canonical_pose1.data, requires_grad=False)
        self.canonical_pose2 = torch.nn.Parameter(canonical_pose2.data, requires_grad=False)
        self.canonical_pose3 = torch.nn.Parameter(canonical_pose3.data, requires_grad=False)
        self.arm_angle_r = torch.nn.Parameter(arm_angle_r.data, requires_grad=True)
        self.arm_angle_l = torch.nn.Parameter(arm_angle_l.data, requires_grad=True)

        self.betas = torch.nn.Parameter(betas.data, requires_grad=False)  # [1, 10]
        self.ground_truth_pose = torch.nn.Parameter(ground_truth_pose.data, requires_grad=False)  # [1, 69]

    def forward(self, x):
        '''
        x are indices telling the dummy estimator what image the currently processed ray is from such that the dummy
        estimator can return the correct expressions... for that ray
        '''

        return torch.cat(
            [self.canonical_pose1, self.arm_angle_l, self.canonical_pose2, self.arm_angle_r, self.canonical_pose3],
            dim=-1), self.betas

    def set_betas(self, betas):
        self.betas = torch.nn.Parameter(betas.data, requires_grad=False)  # [1, 10]

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
