import torch

from models.vertex_sphere_pipeline import VertexSpherePipeline
from solver.smpl_nerf_solver import SmplNerfSolver
from utils import PositionalEncoder


class VertexSphereSolver(SmplNerfSolver):
    def __init__(self, model_coarse, model_fine, positions_encoder: PositionalEncoder,
                 directions_encoder: PositionalEncoder, args,
                 optim=torch.optim.Adam, loss_func=torch.nn.MSELoss()):
        super(SmplNerfSolver, self).__init__(model_coarse, model_fine, positions_encoder, directions_encoder, args,
                                             optim, loss_func)

    def init_pipeline(self):
        return VertexSpherePipeline(self.model_coarse, self.model_fine, self.args, self.positions_encoder,
                                    self.directions_encoder)
