import torch
import torch.nn as nn
import torch.nn.functional as F


class DummySmplEstimatorModel(nn.Module):
    '''
            Uses indices instead of images to fetch preset expressions betas and goal_poses that could be for example perturbed
            Is used by the dynamic_pipeline and dynamic_solver just like the real SmplEstimator
    '''

    def __init__(self, goal_poses, betas):
        super(DummySmplEstimatorModel, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.betas = torch.nn.Parameter(betas.data, requires_grad=False)  # [1, 10]
        self.goal_poses = torch.nn.Parameter(goal_poses.data, requires_grad=False)  # [number_images*h*w, 69]


    def forward(self, x):
        '''
        x are indices telling the dummy estimator what image the currently processed ray is from such that the dummy
        estimator can return the correct expressions... for that ray
        '''
        betas = self.betas.expand(len(x), -1)
        return self.goal_poses[x], betas

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
