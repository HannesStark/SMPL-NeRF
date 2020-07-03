import torch
import numpy as np

from models.smpl_nerf_pipeline import SmplNerfPipeline
from solver.nerf_solver import NerfSolver
from utils import PositionalEncoder, tensorboard_rerenders, tensorboard_warps, GaussianMixture, \
    vedo_data, vedo_data


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
        self.optim = optim(
            list(model_coarse.parameters()) + list(model_fine.parameters()) + list(model_warp_field.parameters()),
            **self.optim_args_merged)

    def init_pipeline(self):
        return SmplNerfPipeline(self.model_coarse, self.model_fine, self.model_warp_field, self.args,
                                self.positions_encoder,
                                self.directions_encoder, self.human_pose_encoder)

    def smpl_nerf_loss(self, rgb, rgb_fine, rgb_truth, warp, densities, ray_samples):
        loss_coarse = self.loss_func(rgb, rgb_truth)
        loss_fine = self.loss_func(rgb_fine, rgb_truth)
        loss = loss_coarse + loss_fine
        if self.args.use_gmm_loss and not self.args.restrict_gmm_loss:
            loss_canonical_densities = self.loss_func(self.canonical_mixture.pdf(ray_samples), densities)
            loss = loss_coarse + loss_fine + loss_canonical_densities
        # loss += 0.5 * torch.mean(torch.norm(warp, p=1, dim=-1))
        return loss, loss_coarse, loss_fine

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
            train_fine_loss = 0
            for i, data in enumerate(train_loader):
                for j, element in enumerate(data):
                    data[j] = element.to(self.device)
                rgb_truth = data[-1]

                rgb, rgb_fine, warp, ray_samples, warped_samples, densities = self.pipeline(data)

                self.optim.zero_grad()
                loss, loss_coarse, loss_fine, = self.smpl_nerf_loss(rgb, rgb_fine, rgb_truth,
                                                                    warp, densities,
                                                                    warped_samples)
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

                            loss, loss_coarse, loss_fine, = self.smpl_nerf_loss(rgb, rgb_fine,
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
            ray_warp_magnitudes = []
            densities_list = []
            image_counter = 0
            for i, data in enumerate(val_loader):
                for j, element in enumerate(data):
                    data[j] = element.to(self.device)
                rgb_truth = data[-1]

                rgb, rgb_fine, warp, ray_samples, warped_samples, densities = self.pipeline(data)

                loss, loss_coarse, loss_fine = self.smpl_nerf_loss(rgb, rgb_fine, rgb_truth, warp, densities,
                                                                   warped_samples)
                val_loss += loss.item()

                ground_truth_images.append(rgb_truth.detach().cpu().numpy())
                rerender_images.append(rgb_fine.detach().cpu().numpy())
                samples.append(ray_samples.detach().cpu().numpy())
                warps.append(warp.detach().cpu().numpy())
                densities_list.append(densities.detach().cpu().numpy())
                warp_magnitude = np.linalg.norm(warp.cpu(), axis=-1)  # [batchsize, number_samples]
                ray_warp_magnitudes.append(warp_magnitude.mean(axis=1))  # mean over the samples => [batchsize]
                if np.concatenate(densities_list).shape[0] >= (h * w):
                    while np.concatenate(densities_list).shape[0]>=(h*w):
                        densities_list = np.concatenate(densities_list)
                        image_densities = densities_list[:h*w].reshape(-1)
                        densities_list = [densities_list[h*w:]]
                        samples = np.concatenate(samples)
                        image_samples = samples[:h*w].reshape(-1, 3)
                        samples = [samples[h*w:]]
                        warps = np.concatenate(warps)
                        image_warps = warps[:h * w].reshape(-1, 3)
                        warps = [samples[h * w:]]
                        print(np.max(image_warps))
                        vedo_data(self.writer, image_densities, image_samples, 
                                  image_warps=image_warps, epoch=epoch + 1,
                                  image_idx=image_counter)
                        image_counter += 1
            if len(val_loader) != 0:
                rerender_images = np.concatenate(rerender_images, 0).reshape((-1, h, w, 3))
                ground_truth_images = np.concatenate(ground_truth_images).reshape((-1, h, w, 3))
                ray_warp_magnitudes = np.concatenate(ray_warp_magnitudes).reshape((-1, h, w))

            tensorboard_rerenders(self.writer, args.number_validation_images, rerender_images, ground_truth_images,
                                  step=epoch + 1, ray_warps=ray_warp_magnitudes)

            print('[Epoch %d] VAL loss: %.7f' % (epoch + 1, val_loss / (len(val_loader) or not len(val_loader))))
            self.writer.add_scalars('Loss Curve', {'train loss': train_loss / iter_per_epoch,
                                                   'val loss': val_loss / (len(val_loader) or not len(val_loader))},
                                    epoch + 1)
            self.writer.add_scalars('Train Losses', {'coarse': train_coarse_loss / iter_per_epoch,
                                                     'fine': train_fine_loss / iter_per_epoch},
                                    epoch + 1)
        print('FINISH.')
