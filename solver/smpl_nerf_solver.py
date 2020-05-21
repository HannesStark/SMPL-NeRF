import torch

from models.smpl_nerf_pipeline import SmplNerfPipeline
from solver.nerf_solver import NerfSolver
from utils import PositionalEncoder


class SmplNerfSolver(NerfSolver):
    def __init__(self, model_coarse, model_fine, model_warp_field, positions_encoder: PositionalEncoder,
                 directions_encoder: PositionalEncoder, human_pose_encoder: PositionalEncoder, args,
                 optim=torch.optim.Adam, loss_func=torch.nn.MSELoss()):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_warp_field = model_warp_field.to(self.device)
        self.human_pose_encoder = human_pose_encoder
        super(SmplNerfSolver, self).__init__(model_coarse, model_fine, positions_encoder, directions_encoder, args,
                                             optim, loss_func)

        self.optim = optim(
            list(model_coarse.parameters()) + list(model_fine.parameters()) + list(model_warp_field.parameters()),
            **self.optim_args_merged)

    def init_pipeline(self):
        return SmplNerfPipeline(self.model_coarse, self.model_fine, self.model_warp_field, self.args,
                                self.positions_encoder,
                                self.directions_encoder, self.human_pose_encoder, self.writer)
