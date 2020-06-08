import torch
import numpy as np

from models.smpl_nerf_pipeline import SmplNerfPipeline
from solver.nerf_solver import NerfSolver
from utils import PositionalEncoder, tensorboard_rerenders, tensorboard_warps, tensorboard_densities, GaussianMixture


class SmplNerfSolver(NerfSolver):
    def __init__(self, model_coarse, model_fine, model_warp_field, positions_encoder: PositionalEncoder,
                 directions_encoder: PositionalEncoder, human_pose_encoder: PositionalEncoder,
                 canonical_smpl, args,
                 optim=torch.optim.Adam, loss_func=torch.nn.MSELoss()):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_warp_field = model_warp_field.to(self.device)
        self.human_pose_encoder = human_pose_encoder
        self.canonical_mixture = GaussianMixture(canonical_smpl, args.gmm_std, self.device)
        super(SmplNerfSolver, self).__init__(model_coarse, model_fine, positions_encoder, directions_encoder, args,
                                             optim, loss_func)
        if args.restrict_gmm_loss:
            self.warp_optim = optim(list(model_coarse.parameters()))
            self.optim = optim(list(model_fine.parameters()) + list(model_warp_field.parameters()),
                **self.optim_args_merged)
        else:
            self.optim = optim(list(model_coarse.parameters()) + list(model_fine.parameters()) + list(model_warp_field.parameters()),
                **self.optim_args_merged)

    def init_pipeline(self):
        return SmplNerfPipeline(self.model_coarse, self.model_fine, self.model_warp_field, self.args,
                                self.positions_encoder,
                                self.directions_encoder, self.human_pose_encoder)

    def smpl_nerf_loss(self, rgb, rgb_fine, rgb_truth, warp, densities, ray_samples):
        loss_coarse = self.loss_func(rgb, rgb_truth)
        loss_fine = self.loss_func(rgb_fine, rgb_truth)
        loss_canonical_densities = self.loss_func(self.canonical_mixture.pdf(ray_samples), densities)
        loss = loss_coarse + loss_fine
        if self.args.use_gmm_loss and not self.args.restrict_gmm_loss:
            loss = loss_coarse + loss_fine + loss_canonical_densities
        # loss += 0.5 * torch.mean(torch.norm(warp, p=1, dim=-1))
        return loss, loss_coarse, loss_fine, loss_canonical_densities

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
            self.model_coarse.train()
            self.model_fine.train()
            train_loss = 0
            train_coarse_loss = 0
            train_densities_loss = 0
            train_fine_loss = 0
            for i, data in enumerate(train_loader):
                for j, element in enumerate(data):
                    data[j] = element.to(self.device)
                rgb_truth = data[-1]

                rgb, rgb_fine, warp, ray_samples, warped_samples, densities = self.pipeline(data)

                self.optim.zero_grad()
                loss, loss_coarse, loss_fine, loss_canonical_densities = self.smpl_nerf_loss(rgb, rgb_fine, rgb_truth,
                                                                                             warp, densities,
                                                                                             warped_samples)
                if self.args.restrict_gmm_loss:
                    loss.backward(retain_graph=True)
                    loss_canonical_densities.backward()
                    self.warp_optim.step()
                else:
                    loss.backward()
                self.optim.step()

                loss_item = loss.item()
                if i % args.log_iterations == args.log_iterations - 1:
                    print('[Epoch %d, Iteration %5d/%5d] TRAIN loss: %.7f' %
                          (epoch + 1, i + 1, iter_per_epoch, loss_item))
                    if args.early_validation:
                        self.model_coarse.eval()
                        self.model_fine.eval()
                        val_loss = 0
                        for j, data in enumerate(val_loader):
                            for j, element in enumerate(data):
                                data[j] = element.to(self.device)
                            rgb_truth = data[-1]

                            rgb, rgb_fine, warp, ray_samples, warped_samples, densities = self.pipeline(data)

                            loss, loss_coarse, loss_fine, loss_canonical_densities = self.smpl_nerf_loss(rgb, rgb_fine,
                                                                                                         rgb_truth,
                                                                                                         warp,
                                                                                                         densities,
                                                                                                         warped_samples)
                            val_loss += loss.item()
                        self.writer.add_scalars('Loss curve every nth iteration', {'train loss': loss_item,
                                                                                   'val loss': val_loss / len(
                                                                                       val_loader)},
                                                i // args.log_iterations + epoch * (
                                                        iter_per_epoch // args.log_iterations))

                train_loss += loss_item
                train_coarse_loss += loss_coarse.item()
                train_densities_loss += loss_canonical_densities.item()
                train_fine_loss += loss_fine.item()
            print('[Epoch %d] Average loss of Epoch: %.7f' %
                  (epoch + 1, train_loss / iter_per_epoch))

            self.model_coarse.eval()
            self.model_fine.eval()
            val_loss = 0
            rerender_images = []
            ground_truth_images = []
            samples = []
            warps = []
            densities_list = []
            warped_samples_list = []
            for i, data in enumerate(val_loader):
                for j, element in enumerate(data):
                    data[j] = element.to(self.device)
                rgb_truth = data[-1]

                rgb, rgb_fine, warp, ray_samples, warped_samples, densities = self.pipeline(data)

                loss, loss_coarse, loss_fine, loss_canonical_densities = self.smpl_nerf_loss(rgb, rgb_fine, rgb_truth,
                                                                                             warp, densities,
                                                                                             warped_samples)
                val_loss += loss.item()

                ground_truth_images.append(rgb_truth.detach().cpu().numpy())
                rerender_images.append(rgb_fine.detach().cpu().numpy())
                samples.append(ray_samples.detach().cpu().numpy())
                warped_samples_list.append(warped_samples.detach().cpu().numpy())
                warps.append(warp.detach().cpu().numpy())
                densities_list.append(densities.detach().cpu().numpy())

            if len(val_loader) != 0:
                rerender_images = np.concatenate(rerender_images, 0).reshape((-1, h, w, 3))
                ground_truth_images = np.concatenate(ground_truth_images).reshape((-1, h, w, 3))
                samples = np.concatenate(samples)
                samples = samples.reshape(
                    (-1, h * w * samples.shape[-2], 3))  # [number_images, h*w*(n_fine_samples + n_coarse_samples), 3]
                warped_samples_list = np.concatenate(warped_samples_list)
                warped_samples_list = warped_samples_list.reshape(
                    (-1, h * w * warped_samples_list.shape[-2],
                     3))  # [number_images, h*w*(n_fine_samples + n_coarse_samples), 3]
                densities_list = np.concatenate(densities_list)
                densities_list = densities_list.reshape(
                    (-1,
                     h * w * densities_list.shape[-1]))  # [number_images, h*w*(n_fine_samples + n_coarse_samples), 3]
                warps = np.concatenate(warps)
                warps_mesh = warps.reshape(
                    (-1, h * w * warps.shape[-2], 3))  # [number_images, h*w*(n_fine_samples + n_coarse_samples), 3]
                warps = warps.reshape((-1, h, w, warps.shape[-2], 3))

            if epoch in np.floor(np.array(args.mesh_epochs) * (args.num_epochs - 1)):  # bc it takes too much storage
                tensorboard_warps(self.writer, args.number_validation_images, samples, warps_mesh, epoch)
                tensorboard_densities(self.writer, args.number_validation_images, warped_samples_list, densities_list,
                                      epoch)

            tensorboard_rerenders(self.writer, args.number_validation_images, rerender_images, ground_truth_images,
                                  step=epoch, warps=warps)

            print('[Epoch %d] VAL loss: %.7f' % (epoch + 1, val_loss / (len(val_loader) or not len(val_loader))))
            self.writer.add_scalars('Loss Curve', {'train loss': train_loss / iter_per_epoch,
                                                   'val loss': val_loss / (len(val_loader) or not len(val_loader))},
                                    epoch)
            self.writer.add_scalars('Train Losses', {'coarse': train_coarse_loss / iter_per_epoch,
                                                     'fine': train_fine_loss / iter_per_epoch,
                                                     'densities': train_densities_loss / iter_per_epoch},
                                    epoch)
        print('FINISH.')
