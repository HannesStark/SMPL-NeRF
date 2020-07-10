import torch
import numpy as np

from models.dynamic_pipeline import DynamicPipeline
from solver.nerf_solver import NerfSolver
from utils import PositionalEncoder, tensorboard_rerenders, vedo_data


class DynamicSolver(NerfSolver):
    '''
    Solver for the full pipeline with smpl estimator that produces an smpl that is used to calculate the warp
    which is added to each sample that is then passed to the NeRF to get an output. The loss of the output is taken
    and backpropagated to optimize the NeRF and the smpl estimator.
    '''

    def __init__(self, model_coarse, model_fine, smpl_estimator, smpl_model, positions_encoder: PositionalEncoder,
                 directions_encoder: PositionalEncoder, args,
                 optim=torch.optim.Adam, loss_func=torch.nn.MSELoss()):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.smpl_estimator = smpl_estimator.to(self.device)
        self.smpl_model = smpl_model.to(self.device)
        super(DynamicSolver, self).__init__(model_coarse, model_fine, positions_encoder, directions_encoder, args,
                                            optim, loss_func)
        self.optim = optim(
            list(model_coarse.parameters()) + list(model_fine.parameters()),
            **self.optim_args_merged)

    def init_pipeline(self):
        return DynamicPipeline(self.model_coarse, self.model_fine, self.smpl_estimator, self.smpl_model,
                               self.args,
                               self.positions_encoder,
                               self.directions_encoder)

    def loss(self, rgb, rgb_fine, rgb_truth, warp, densities, ray_samples):
        loss_coarse = self.loss_func(rgb, rgb_truth)
        loss_fine = self.loss_func(rgb_fine, rgb_truth)
        loss = loss_coarse + loss_fine
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
            self.smpl_estimator.train()
            train_loss = 0
            train_coarse_loss = 0
            train_fine_loss = 0
            for i, data in enumerate(train_loader):
                for j, element in enumerate(data):
                    data[j] = element.to(self.device)
                rgb_truth = data[-1]

                rgb, rgb_fine, warp, ray_samples, warped_samples, densities = self.pipeline(data)

                self.optim.zero_grad()
                loss, loss_coarse, loss_fine, = self.loss(rgb, rgb_fine, rgb_truth,
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

                            loss, loss_coarse, loss_fine, = self.loss(rgb, rgb_fine,
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
            self.smpl_estimator.eval()
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

                loss, loss_coarse, loss_fine = self.loss(rgb, rgb_fine, rgb_truth, warp, densities,
                                                         warped_samples)
                val_loss += loss.item()

                ground_truth_images.append(rgb_truth.detach().cpu().numpy())
                rerender_images.append(rgb_fine.detach().cpu().numpy())
                samples.append(ray_samples.detach().cpu().numpy())
                warps.append(warp.detach().cpu().numpy())
                densities_list.append(densities.detach().cpu().numpy())
                warp_magnitude = np.linalg.norm(warp.detach().cpu(), axis=-1)  # [batchsize, number_samples]
                ray_warp_magnitudes.append(warp_magnitude.mean(axis=1))  # mean over the samples => [batchsize]
                if np.concatenate(densities_list).shape[0] >= (h * w):
                    while np.concatenate(densities_list).shape[0] >= (h * w):
                        densities_list = np.concatenate(densities_list)
                        image_densities = densities_list[:h * w].reshape(-1)
                        densities_list = [densities_list[h * w:]]
                        samples = np.concatenate(samples)
                        image_samples = samples[:h * w].reshape(-1, 3)
                        samples = [samples[h * w:]]
                        warps = np.concatenate(warps)
                        image_warps = warps[:h * w].reshape(-1, 3)
                        warps = [warps[h * w:]]
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
