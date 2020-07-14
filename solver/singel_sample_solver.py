import torch

from models.singe_sample_pipeline import SmplPipeline
from solver.nerf_solver import NerfSolver
from utils import PositionalEncoder


class SmplSolver(NerfSolver):
    def __init__(self, model_coarse, model_fine, positions_encoder: PositionalEncoder,
                 directions_encoder: PositionalEncoder, args,
                 optim=torch.optim.Adam, loss_func=torch.nn.MSELoss()):
        super(SmplSolver, self).__init__(model_coarse, model_fine, positions_encoder, directions_encoder, args,
                                         optim, loss_func)

    def init_pipeline(self):
        return SmplPipeline(self.model_coarse, self.args, self.positions_encoder, self.directions_encoder)
