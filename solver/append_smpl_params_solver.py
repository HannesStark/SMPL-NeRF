import torch

from models.append_smpl_params_pipeline import AppendSmplParamsPipeline
from solver.nerf_solver import NerfSolver
from utils import PositionalEncoder


class AppendSmplParamsSolver(NerfSolver):
    def __init__(self, model_coarse, model_fine, positions_encoder: PositionalEncoder,
                 directions_encoder: PositionalEncoder, human_pose_encoder: PositionalEncoder, args,
                 optim=torch.optim.Adam, loss_func=torch.nn.MSELoss()):
        self.human_pose_encoder = human_pose_encoder
        super(AppendSmplParamsSolver, self).__init__(model_coarse, model_fine, positions_encoder, directions_encoder, args,
                                                 optim, loss_func)

    def init_pipeline(self):
        return AppendSmplParamsPipeline(self.model_coarse, self.model_fine, self.args,
                                    self.positions_encoder,
                                    self.directions_encoder, self.human_pose_encoder)
