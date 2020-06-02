import torch
import numpy as np

from models.append_to_nerf_pipeline import AppendToNerfPipeline
from models.smpl_nerf_pipeline import SmplNerfPipeline
from solver.nerf_solver import NerfSolver
from utils import PositionalEncoder, tensorboard_rerenders, tensorboard_warps, tensorboard_densities, GaussianMixture


class AppendToNerfSolver(NerfSolver):
    def __init__(self, model_coarse, model_fine, positions_encoder: PositionalEncoder,
                 directions_encoder: PositionalEncoder, human_pose_encoder: PositionalEncoder, args,
                 optim=torch.optim.Adam, loss_func=torch.nn.MSELoss()):
        self.human_pose_encoder = human_pose_encoder
        super(AppendToNerfSolver, self).__init__(model_coarse, model_fine, positions_encoder, directions_encoder, args,
                                             optim, loss_func)

    def init_pipeline(self):
        return AppendToNerfPipeline(self.model_coarse, self.model_fine, self.args,
                                    self.positions_encoder,
                                    self.directions_encoder, self.human_pose_encoder, self.writer)
