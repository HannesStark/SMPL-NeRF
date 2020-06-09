import torch
import numpy as np

from solver.nerf_solver import NerfSolver
from utils import PositionalEncoder, tensorboard_rerenders, tensorboard_warps, tensorboard_densities, GaussianMixture
from torch.utils.tensorboard import SummaryWriter


class WarpSolver(NerfSolver):
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0}

    def __init__(self, model_warp_field, positions_encoder: PositionalEncoder,
                 directions_encoder: PositionalEncoder, human_pose_encoder: PositionalEncoder,
                 args, optim=torch.optim.Adam, loss_func=torch.nn.MSELoss()):
        self.position_encoder = positions_encoder
        self.direction_encoder = directions_encoder
        self.optim_args_merged = self.default_adam_args.copy()
        self.optim_args_merged.update({"lr": args.lrate, "weight_decay": args.weight_decay})
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_warp_field = model_warp_field.to(self.device)
        self.human_pose_encoder = human_pose_encoder
        self.args = args
        self.writer = SummaryWriter()
        self.optim = optim(list(model_warp_field.parameters()),
                           **self.optim_args_merged)
        self.loss_func = loss_func

    def forward(self, ray_sample, goal_pose):
        goal_pose = torch.stack([goal_pose[:, 38], goal_pose[:, 41]], axis=-1)
        # get values for coarse network and run them through the coarse network
        goal_pose_encoding = self.human_pose_encoder.encode(goal_pose)
        samples_encoding = self.position_encoder.encode(ray_sample)
        if self.args.human_pose_encoding:
            warp_field_inputs = torch.cat([samples_encoding,
                                           goal_pose_encoding], -1)
        else:
            warp_field_inputs = torch.cat(
                [ray_sample, goal_pose], -1)
        warp = self.model_warp_field(warp_field_inputs)
        return warp

    def train(self, train_loader, val_loader, h: int, w: int):
        """
        Train coarse and fine model on training data and run validation

        Parameters
        ----------
        train_loader : training data loader object.
        val_loader : validation data loader object.
        h : int
            height of images.
        w : int
            width of images.
        """
        args = self.args
        iter_per_epoch = len(train_loader)

        print('START TRAIN.')

        for epoch in range(args.num_epochs):  # loop over the dataset multiple times
            ### Training ###
            train_loss = 0
            for i, data in enumerate(train_loader):
                for j, element in enumerate(data):
                    data[j] = element.to(self.device)
                ray_sample, ray_translation, samples_direction, warp_truth, rgb_truth, goal_pose = data
                warp = self.forward(ray_sample, goal_pose)
                warped_samples = ray_sample + warp
                self.optim.zero_grad()
                loss = self.loss_func(warp, warp_truth)
                loss.backward()
                loss_item = loss.item()
                self.optim.step()
                if i % args.log_iterations == args.log_iterations - 1:
                    print('[Epoch %d, Iteration %5d/%5d] TRAIN loss: %.7f' %
                          (epoch + 1, i + 1, iter_per_epoch, loss_item))
                train_loss += loss_item
            print('[Epoch %d] Average loss of Epoch: %.7f' %
                  (epoch + 1, train_loss / iter_per_epoch))

            ### Validation ###
            val_loss = 0
            ground_truth_images = []
            samples = []
            warps = []
            ground_truth_warps = []
            densities_list = []
            warped_samples_list = []
            for i, data in enumerate(val_loader):
                for j, element in enumerate(data):
                    data[j] = element.to(self.device)
                ray_sample, ray_translation, samples_direction, warp_truth, rgb_truth, goal_pose = data
                warp = self.forward(ray_sample, goal_pose)
                warped_samples = ray_sample + warp
                loss = self.loss_func(warp, warp_truth)
                val_loss += loss.item()

                ground_truth_images.append(rgb_truth.detach().cpu().numpy())
                samples.append(ray_sample.detach().cpu().numpy())
                warped_samples_list.append(warped_samples.detach().cpu().numpy())
                warps.append(warp.detach().cpu().numpy())
                ground_truth_warps.append(warp_truth.detach().cpu().numpy())

            if len(val_loader) != 0:
                ground_truth_images = np.concatenate(ground_truth_images).reshape((-1, h, w, 3))
                samples = np.concatenate(samples)
                samples = samples.reshape(
                    (-1, h * w, 3))  # [number_images, h*w*(n_fine_samples + n_coarse_samples), 3]
                warped_samples_list = np.concatenate(warped_samples_list)
                warped_samples_list = warped_samples_list.reshape(
                    (-1, h * w,
                     3))  # [number_images, h*w*(n_fine_samples + n_coarse_samples), 3]
                warps = np.concatenate(warps)
                warps_mesh = warps.reshape(
                    (-1, h * w, 3))  # [number_images, h*w*(n_fine_samples + n_coarse_samples), 3]
                warps = warps.reshape((-1, h, w, 3))
                
                ground_truth_warps = np.concatenate(ground_truth_warps)
                ground_truth_warps_mesh = ground_truth_warps.reshape(
                    (-1, h * w, 3))  # [number_images, h*w*(n_fine_samples + n_coarse_samples), 3]
                ground_truth_warps = ground_truth_warps.reshape((-1, h, w, 3))
            if epoch in np.floor(np.array(args.mesh_epochs) * (args.num_epochs - 1)):  # bc it takes too much storage
                tensorboard_warps(self.writer, args.number_validation_images, samples, warps_mesh, epoch, point_size=h/1000)
                tensorboard_warps(self.writer, args.number_validation_images, 
                                  samples, ground_truth_warps_mesh, epoch, tensorboard_tag="warp_gt", point_size=h/1000)

            print('[Epoch %d] VAL loss: %.7f' % (epoch + 1, val_loss / (len(val_loader) or not len(val_loader))))
            self.writer.add_scalars('Loss Curve', {'train loss': train_loss / iter_per_epoch,
                                                   'val loss': val_loss / (len(val_loader) or not len(val_loader))},
                                    epoch)
        print('FINISH.')
