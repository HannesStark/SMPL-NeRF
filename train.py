import os

import smplx
import torch
from torch.utils.data import Subset
from torchvision.transforms import transforms
from config_parser import config_parser
from datasets.dummy_dynamic_dataset import DummyDynamicDataset
from datasets.image_wise_dataset import ImageWiseDataset
from datasets.rays_from_images_dataset import RaysFromImagesDataset
from datasets.single_sample_dataset import SmplDataset
from datasets.smpl_nerf_dataset import SmplNerfDataset
from datasets.smpl_estimator_dataset import SmplEstimatorDataset
from datasets.transforms import CoarseSampling, ToTensor, NormalizeRGB, NormalizeRGBImage
from datasets.vertex_sphere_dataset import VertexSphereDataset
from datasets.original_nerf_dataset import OriginalNerfDataset
from models.append_vertices_net import AppendVerticesNet
from models.debug_model import DebugModel
from models.siren_net import SirenRenderRayNet
from models.dummy_image_wise_estimator import DummyImageWiseEstimator
from models.dummy_smpl_estimator_model import DummySmplEstimatorModel
from models.render_ray_net import RenderRayNet
from models.warp_field_net import WarpFieldNet
from solver.append_smpl_params_solver import AppendSmplParamsSolver
from solver.append_vertices_solver import AppendVerticesSolver
from solver.dynamic_solver import DynamicSolver
from solver.image_wise_solver import ImageWiseSolver
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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.default_device = device
    if args.model_type not in ["nerf", "smpl_nerf", "append_to_nerf", "smpl", "warp", 'vertex_sphere', "smpl_estimator",
                               "original_nerf", 'dummy_dynamic', 'image_wise_dynamic',
                               "append_vertex_locations_to_nerf", 'append_smpl_params']:
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
    elif args.model_type == "smpl_nerf" or args.model_type == "append_to_nerf" or args.model_type == "append_smpl_params":
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
    elif args.model_type == "append_vertex_locations_to_nerf":
        train_data = DummyDynamicDataset(train_dir, os.path.join(train_dir, 'transforms.json'), transform)
        val_data = DummyDynamicDataset(val_dir, os.path.join(val_dir, 'transforms.json'), transform)
    elif args.model_type == 'image_wise_dynamic':
        canonical_pose1 = torch.zeros(38).view(1, -1)
        canonical_pose2 = torch.zeros(2).view(1, -1)
        canonical_pose3 = torch.zeros(27).view(1, -1)
        arm_angle_l = torch.tensor([np.deg2rad(10)]).float().view(1, -1)
        arm_angle_r = torch.tensor([np.deg2rad(10)]).float().view(1, -1)
        smpl_estimator = DummyImageWiseEstimator(canonical_pose1, canonical_pose2, canonical_pose3, arm_angle_l,
                                                 arm_angle_r, torch.zeros(10).view(1, -1), torch.zeros(69).view(1, -1))
        train_data = ImageWiseDataset(train_dir, os.path.join(train_dir, 'transforms.json'), smpl_estimator,
                                      transform,
                                      args)
        val_data = ImageWiseDataset(val_dir, os.path.join(val_dir, 'transforms.json'), smpl_estimator, transform, args)
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

    elif args.model_type == 'smpl':
        solver = SmplSolver(model_coarse, model_fine, position_encoder, direction_encoder,
                            args, torch.optim.Adam,
                            torch.nn.MSELoss())
        solver.train(train_loader, val_loader, train_data.h, train_data.w, parser)
        save_run(solver.writer.log_dir, [model_coarse, model_fine],
                 ['model_coarse.pt', 'model_fine.pt'], parser)

    elif args.model_type == 'nerf' or args.model_type == "original_nerf":
        solver = NerfSolver(model_coarse, model_fine, position_encoder, direction_encoder, args, torch.optim.Adam,
                            torch.nn.MSELoss())
        solver.train(train_loader, val_loader, train_data.h, train_data.w, parser)
        save_run(solver.writer.log_dir, [model_coarse, model_fine],
                 ['model_coarse.pt', 'model_fine.pt'], parser)

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
    elif args.model_type == 'append_smpl_params':
        human_pose_encoder = PositionalEncoder(args.number_frequencies_pose, args.use_identity_pose)
        human_pose_dim = human_pose_encoder.output_dim if args.human_pose_encoding else 1

        model_coarse = RenderRayNet(args.netdepth, args.netwidth, position_encoder.output_dim * 3,
                                    direction_encoder.output_dim * 3, human_pose_dim * 69,
                                    skips=args.skips, use_directional_input=args.use_directional_input)
        model_fine = RenderRayNet(args.netdepth_fine, args.netwidth_fine, position_encoder.output_dim * 3,
                                  direction_encoder.output_dim * 3, human_pose_dim * 69,
                                  skips=args.skips_fine, use_directional_input=args.use_directional_input)

        if args.load_run is not None:
            model_coarse.load_state_dict(
                torch.load(os.path.join(args.load_run, 'model_coarse.pt'), map_location=torch.device(device)))
            model_fine.load_state_dict(
                torch.load(os.path.join(args.load_run, 'model_fine.pt'), map_location=torch.device(device)))
            print("Models loaded from ", args.load_run)
        if args.siren:
            model_coarse = SirenRenderRayNet(args.netdepth, args.netwidth, position_encoder.output_dim * 3,
                                        direction_encoder.output_dim * 3, human_pose_dim * 69,
                                        skips=args.skips, use_directional_input=args.use_directional_input)
            model_fine = SirenRenderRayNet(args.netdepth_fine, args.netwidth_fine, position_encoder.output_dim * 3,
                                      direction_encoder.output_dim * 3, human_pose_dim * 69,
                                      skips=args.skips_fine, use_directional_input=args.use_directional_input)
        solver = AppendSmplParamsSolver(model_coarse, model_fine, position_encoder, direction_encoder, human_pose_encoder,
                                    args, torch.optim.Adam,
                                    torch.nn.MSELoss())
        solver.train(train_loader, val_loader, train_data.h, train_data.w, parser)

        save_run(solver.writer.log_dir, [model_coarse, model_fine],
                 ['model_coarse.pt', 'model_fine.pt'], parser)

        model_dependent = [human_pose_encoder, human_pose_dim]
        inference_gif(solver.writer.log_dir, args.model_type, args, train_data, val_data, position_encoder,
                      direction_encoder, model_coarse, model_fine, model_dependent)
    elif args.model_type == 'append_to_nerf':
        human_pose_encoder = PositionalEncoder(args.number_frequencies_pose, args.use_identity_pose)
        human_pose_dim = human_pose_encoder.output_dim if args.human_pose_encoding else 1
        model_coarse = RenderRayNet(args.netdepth, args.netwidth, position_encoder.output_dim * 3,
                                    direction_encoder.output_dim * 3, human_pose_dim * 2,
                                    skips=args.skips, use_directional_input=args.use_directional_input)
        model_fine = RenderRayNet(args.netdepth_fine, args.netwidth_fine, position_encoder.output_dim * 3,
                                  direction_encoder.output_dim * 3, human_pose_dim * 2,
                                  skips=args.skips_fine, use_directional_input=args.use_directional_input)
        solver = AppendToNerfSolver(model_coarse, model_fine, position_encoder, direction_encoder, human_pose_encoder,
                                    args, torch.optim.Adam,
                                    torch.nn.MSELoss())
        solver.train(train_loader, val_loader, train_data.h, train_data.w,   parser)

        save_run(solver.writer.log_dir, [model_coarse, model_fine],
                 ['model_coarse.pt', 'model_fine.pt'], parser)

        model_dependent = [human_pose_encoder, human_pose_dim]
        inference_gif(solver.writer.log_dir, args.model_type, args, train_data, val_data, position_encoder,
                      direction_encoder, model_coarse, model_fine, model_dependent)
    elif args.model_type == 'append_vertex_locations_to_nerf':
        model_coarse = AppendVerticesNet(args.netdepth, args.netwidth, position_encoder.output_dim * 3,
                                         direction_encoder.output_dim * 3, 6890, additional_input_layers=1,
                                         skips=args.skips)
        model_fine = AppendVerticesNet(args.netdepth_fine, args.netwidth_fine, position_encoder.output_dim * 3,
                                       direction_encoder.output_dim * 3, 6890, additional_input_layers=1,
                                       skips=args.skips_fine)
        smpl_estimator = DummySmplEstimatorModel(train_data.goal_poses, train_data.betas)
        smpl_file_name = "SMPLs/smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl"
        smpl_model = smplx.create(smpl_file_name, model_type='smpl')
        smpl_model.batchsize = args.batchsize
        solver = AppendVerticesSolver(model_coarse, model_fine, smpl_estimator, smpl_model, position_encoder,
                                      direction_encoder,
                                      args, torch.optim.Adam,
                                      torch.nn.MSELoss())
        solver.train(train_loader, val_loader, train_data.h, train_data.w)

        save_run(solver.writer.log_dir, [model_coarse, model_fine],
                 ['model_coarse.pt', 'model_fine.pt'], parser)

    elif args.model_type == 'vertex_sphere':
        solver = VertexSphereSolver(model_coarse, model_fine, position_encoder, direction_encoder, args,
                                    torch.optim.Adam,
                                    torch.nn.MSELoss())
        solver.train(train_loader, val_loader, train_data.h, train_data.w)
        save_run(solver.writer.log_dir, [model_coarse, model_fine],
                 ['model_coarse.pt', 'model_fine.pt'], parser)

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
        smpl_estimator = DummySmplEstimatorModel(train_data.goal_poses, train_data.betas)
        solver = DynamicSolver(model_fine, model_coarse, smpl_estimator, smpl_model, position_encoder,
                               direction_encoder, args)
        solver.train(train_loader, val_loader, train_data.h, train_data.w)
        save_run(solver.writer.log_dir, [model_coarse, model_fine, smpl_estimator],
                 ['model_coarse.pt', 'model_fine.pt', 'smpl_estimator.pt'], parser)
    elif args.model_type == "image_wise_dynamic":
        if args.load_coarse_model != None:
            print("Load model..")
            model_coarse.load_state_dict(
                torch.load(args.load_coarse_model, map_location=torch.device(device)))
            for params in model_coarse.parameters():
                params.requires_grad = False
            model_coarse.eval()
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True, num_workers=0)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False, num_workers=0)
        smpl_file_name = "SMPLs/smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl"
        smpl_model = smplx.create(smpl_file_name, model_type='smpl')
        smpl_model.batchsize = args.batchsize
        solver = ImageWiseSolver(model_coarse, model_fine, smpl_estimator, smpl_model, position_encoder,
                                 direction_encoder, args)
        solver.train(train_loader, val_loader, train_data.h, train_data.w)
        save_run(solver.writer.log_dir, [model_coarse, model_fine, smpl_estimator],
                 ['model_coarse.pt', 'model_fine.pt', 'smpl_estimator.pt'], parser)


if __name__ == '__main__':
    train()
