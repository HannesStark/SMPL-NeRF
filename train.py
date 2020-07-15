import os

import smplx
import torch
from torch.utils.data import Subset
from torchvision.transforms import transforms
from config_parser import config_parser
from datasets.dummy_dynamic_dataset import DummyDynamicDataset
from datasets.rays_from_images_dataset import RaysFromImagesDataset
from datasets.single_sample_dataset import SmplDataset
from datasets.smpl_nerf_dataset import SmplNerfDataset
from datasets.smpl_estimator_dataset import SmplEstimatorDataset
from datasets.transforms import CoarseSampling, ToTensor, NormalizeRGB, NormalizeRGBImage
from datasets.vertex_sphere_dataset import VertexSphereDataset
from datasets.original_nerf_dataset import OriginalNerfDataset
from models.debug_model import DebugModel
from models.dummy_smpl_estimator_model import DummySmplEstimatorModel
from models.render_ray_net import RenderRayNet
from models.warp_field_net import WarpFieldNet
from solver.dynamic_solver import DynamicSolver
from solver.vertex_sphere_solver import VertexSphereSolver
from solver.append_to_nerf_solver import AppendToNerfSolver
from solver.nerf_solver import NerfSolver
from solver.warp_solver import WarpSolver
import numpy as np

from solver.smpl_nerf_solver import SmplNerfSolver
from solver.singel_sample_solver import SmplSolver
from utils import PositionalEncoder, save_run
from models.smpl_estimator import SmplEstimator
from solver.smpl_estimator_solver import SmplEstimatorSolver
from inference import inference_gif

np.random.seed(0)


def train():
    parser = config_parser()
    args = parser.parse_args()
    if args.model_type not in ["nerf", "smpl_nerf", "append_to_nerf", "smpl", "warp", 'vertex_sphere', "smpl_estimator",
                               "original_nerf", 'dummy_dynamic']:
        raise Exception("The model type ", args.model_type, " does not exist.")

    transform = transforms.Compose(
        [NormalizeRGB(), CoarseSampling(args.near, args.far, args.number_coarse_samples), ToTensor()])

    train_dir = os.path.join(args.dataset_dir, 'train')
    val_dir = os.path.join(args.dataset_dir, 'val')
    if args.model_type == "nerf":
        train_data = RaysFromImagesDataset(train_dir, os.path.join(train_dir, 'transforms.json'), transform)
        val_data = RaysFromImagesDataset(val_dir, os.path.join(val_dir, 'transforms.json'), transform)
    elif args.model_type == "smpl" or args.model_type == "warp":
        train_data = SmplDataset(train_dir, os.path.join(train_dir, 'transforms.json'), args, transform=NormalizeRGB())
        val_data = SmplDataset(val_dir, os.path.join(val_dir, 'transforms.json'), args, transform=NormalizeRGB())
    elif args.model_type == "smpl_nerf" or args.model_type == "append_to_nerf":
        train_data = SmplNerfDataset(train_dir, os.path.join(train_dir, 'transforms.json'), transform)
        val_data = SmplNerfDataset(val_dir, os.path.join(val_dir, 'transforms.json'), transform)
    elif args.model_type == "vertex_sphere":
        train_data = VertexSphereDataset(train_dir, os.path.join(train_dir, 'transforms.json'), args)
        val_data = VertexSphereDataset(val_dir, os.path.join(val_dir, 'transforms.json'), args)
    elif args.model_type == "smpl_estimator":
        transform = NormalizeRGBImage()
        train_data = SmplEstimatorDataset(train_dir, os.path.join(train_dir, 'transforms.json'),
                                          args.vertex_sphere_radius, transform)
        val_data = SmplEstimatorDataset(val_dir, os.path.join(val_dir, 'transforms.json'), args.vertex_sphere_radius,
                                        transform)
    elif args.model_type == "original_nerf":
        train_data = OriginalNerfDataset(args.dataset_dir, os.path.join(args.dataset_dir, 'transforms_train.json'),
                                         transform)
        val_data = OriginalNerfDataset(args.dataset_dir, os.path.join(args.dataset_dir, 'transforms_val.json'),
                                       transform)
    elif args.model_type == "dummy_dynamic":
        train_data = DummyDynamicDataset(train_dir, os.path.join(train_dir, 'transforms.json'), transform)
        val_data = DummyDynamicDataset(val_dir, os.path.join(val_dir, 'transforms.json'), transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batchsize, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batchsize_val, shuffle=False, num_workers=0)
    position_encoder = PositionalEncoder(args.number_frequencies_postitional, args.use_identity_positional)
    direction_encoder = PositionalEncoder(args.number_frequencies_directional, args.use_identity_directional)
    model_coarse = RenderRayNet(args.netdepth, args.netwidth, position_encoder.output_dim * 3,
                                direction_encoder.output_dim * 3, skips=args.skips)
    model_fine = RenderRayNet(args.netdepth_fine, args.netwidth_fine, position_encoder.output_dim * 3,
                              direction_encoder.output_dim * 3, skips=args.skips_fine)

    if args.model_type == "smpl_nerf":
        human_pose_encoder = PositionalEncoder(args.number_frequencies_pose, args.use_identity_pose)
        positions_dim = position_encoder.output_dim if args.human_pose_encoding else 1
        human_pose_dim = human_pose_encoder.output_dim if args.human_pose_encoding else 1
        model_warp_field = WarpFieldNet(args.netdepth_warp, args.netwidth_warp, positions_dim * 3,
                                        human_pose_dim * 2)

        solver = SmplNerfSolver(model_coarse, model_fine, model_warp_field, position_encoder, direction_encoder,
                                human_pose_encoder, train_data.canonical_smpl, args, torch.optim.Adam,
                                torch.nn.MSELoss())
        solver.train(train_loader, val_loader, train_data.h, train_data.w)

        save_run(solver.writer.log_dir, [model_coarse, model_fine, model_warp_field],
                 ['model_coarse.pt', 'model_fine.pt', 'model_warp_field.pt'], parser)

        model_dependent = [human_pose_encoder, positions_dim, human_pose_dim, model_warp_field]
        #inference_gif(solver.writer.log_dir, args.model_type, args, train_data, val_data, position_encoder,
        #              direction_encoder, model_coarse, model_fine, model_dependent)

    elif args.model_type == 'smpl':
        solver = SmplSolver(model_coarse, model_fine, position_encoder, direction_encoder,
                            args, torch.optim.Adam,
                            torch.nn.MSELoss())
        solver.train(train_loader, val_loader, train_data.h, train_data.w)
        save_run(solver.writer.log_dir, [model_coarse, model_fine],
                 ['model_coarse.pt', 'model_fine.pt'], parser)
        #inference_gif(solver.writer.log_dir, args.model_type, args, train_data, val_data, position_encoder,
        #              direction_encoder, model_coarse, model_fine, [])

    elif args.model_type == 'nerf' or args.model_type == "original_nerf":
        solver = NerfSolver(model_coarse, model_fine, position_encoder, direction_encoder, args, torch.optim.Adam,
                            torch.nn.MSELoss())
        solver.train(train_loader, val_loader, train_data.h, train_data.w)
        save_run(solver.writer.log_dir, [model_coarse, model_fine],
                 ['model_coarse.pt', 'model_fine.pt'], parser)
        #inference_gif(solver.writer.log_dir, args.model_type, args, train_data, val_data, position_encoder,
        #              direction_encoder, model_coarse, model_fine, [])

    elif args.model_type == 'warp':
        human_pose_encoder = PositionalEncoder(args.number_frequencies_pose, args.use_identity_pose)
        positions_dim = position_encoder.output_dim if args.human_pose_encoding else 1
        human_pose_dim = human_pose_encoder.output_dim if args.human_pose_encoding else 1
        model_warp_field = WarpFieldNet(args.netdepth_warp, args.netwidth_warp, positions_dim * 3,
                                        human_pose_dim * 2)
        human_pose_encoder = PositionalEncoder(args.number_frequencies_pose, args.use_identity_pose)
        solver = WarpSolver(model_warp_field, position_encoder, direction_encoder, human_pose_encoder, args)
        solver.train(train_loader, val_loader, train_data.h, train_data.w)
        save_run(solver.writer.log_dir, [model_warp_field],
                 ['model_warp_field.pt'], parser)
    elif args.model_type == 'append_to_nerf':
        human_pose_encoder = PositionalEncoder(args.number_frequencies_pose, args.use_identity_pose)
        human_pose_dim = human_pose_encoder.output_dim if args.human_pose_encoding else 1
        model_coarse = RenderRayNet(args.netdepth, args.netwidth, position_encoder.output_dim * 3,
                                    direction_encoder.output_dim * 3, human_pose_dim * 2,
                                    skips=args.skips)
        model_fine = RenderRayNet(args.netdepth_fine, args.netwidth_fine, position_encoder.output_dim * 3,
                                  direction_encoder.output_dim * 3, human_pose_dim * 2,
                                  skips=args.skips_fine)
        solver = AppendToNerfSolver(model_coarse, model_fine, position_encoder, direction_encoder, human_pose_encoder,
                                    args, torch.optim.Adam,
                                    torch.nn.MSELoss())
        solver.train(train_loader, val_loader, train_data.h, train_data.w)

        save_run(solver.writer.log_dir, [model_coarse, model_fine],
                 ['model_coarse.pt', 'model_fine.pt'], parser)

        model_dependent = [human_pose_encoder, human_pose_dim]
        inference_gif(solver.writer.log_dir, args.model_type, args, train_data, val_data, position_encoder,
                      direction_encoder, model_coarse, model_fine, model_dependent)

    elif args.model_type == 'vertex_sphere':
        solver = VertexSphereSolver(model_coarse, model_fine, position_encoder, direction_encoder, args,
                                    torch.optim.Adam,
                                    torch.nn.MSELoss())
        solver.train(train_loader, val_loader, train_data.h, train_data.w)
        save_run(solver.writer.log_dir, [model_coarse, model_fine],
                 ['model_coarse.pt', 'model_fine.pt'], parser)
        #inference_gif(solver.writer.log_dir, args.model_type, args, train_data, val_data, position_encoder,
        #              direction_encoder, model_coarse, model_fine, [])

    elif args.model_type == 'smpl_estimator':

        model = SmplEstimator(human_size=len(args.human_joints))

        solver = SmplEstimatorSolver(model, args, torch.optim.Adam,
                                     torch.nn.MSELoss())
        solver.train(train_loader, val_loader)
        save_run(solver.writer.log_dir, [model],
                 ['model_smpl_estimator.pt'], parser)
    elif args.model_type == "dummy_dynamic":
        smpl_file_name = "SMPLs/smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl"
        smpl_model = smplx.create(smpl_file_name, model_type='smpl')
        smpl_model.batchsize = args.batchsize
        smpl_estimator = DummySmplEstimatorModel(train_data.goal_poses, train_data.betas, train_data.expression)
        parameters = smpl_estimator.parameters()
        solver = DynamicSolver(model_fine, model_coarse, smpl_estimator, smpl_model, position_encoder,
                               direction_encoder, args)
        solver.train(train_loader, val_loader, train_data.h, train_data.w)
        save_run(solver.writer.log_dir, [model_coarse, model_fine, smpl_estimator],
                 ['model_coarse.pt', 'model_fine.pt', 'smpl_estimator.pt'], parser)


if __name__ == '__main__':
    train()
