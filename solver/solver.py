import torch
from torch.utils.tensorboard import SummaryWriter

from utils import run_nerf_pipeline, PositionalEncoder


class Solver():
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0}

    def __init__(self, positions_encoder: PositionalEncoder, directions_encoder: PositionalEncoder,
                 optim=torch.optim.Adam, optim_args={},
                 loss_func=torch.nn.MSELoss(), sigma_noise_std=1,
                 white_background=False, number_fine_samples=128):
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func
        self.sigma_noise_std = sigma_noise_std
        self.white_background = white_background
        self.positions_encoder = positions_encoder
        self.directions_encoder = directions_encoder
        self.number_fine_samples = number_fine_samples
        self.writer = SummaryWriter()
        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_loss_history_per_iter = []
        self.val_loss_history = []

    def train(self, model_coarse, model_fine, train_loader, val_loader, num_epochs=10, log_nth=0):
        """
        Train a given model with the provided data.

        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        - num_plots: Number of plots that will be created for tensorboard (is batchsize if batchsize is lower)
        """

        optim = self.optim(list(model_coarse.parameters()) + list(model_fine.parameters()), **self.optim_args)
        self._reset_histories()
        iter_per_epoch = len(train_loader)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device = 'cpu'
        # if torch.cuda.is_available():
        #    torch.set_default_tensor_type('torch.cuda.FloatTensor')
        model_coarse.to(device)
        model_fine.to(device)

        print('START TRAIN.')

        for epoch in range(num_epochs):  # loop over the dataset multiple times
            model_coarse.train()
            model_fine.train()
            train_loss = 0
            for i, data in enumerate(train_loader):
                ray_samples, ray_translation, ray_direction, z_vals, rgb_truth = data
                ray_samples = ray_samples.to(device)  # [batchsize, number_coarse_samples, 3]
                ray_translation = ray_translation.to(device)  # [batchsize, 3]
                ray_direction = ray_direction.to(device)  # [batchsize, 3]
                z_vals = z_vals.to(device)  # [batchsize, number_coarse_samples]
                rgb_truth = rgb_truth.to(device)  # [batchsize, 3]

                rgb, rgb_fine = run_nerf_pipeline(ray_samples, ray_translation, ray_direction, z_vals,
                                                  model_coarse, model_fine, self.sigma_noise_std,
                                                  self.number_fine_samples, self.white_background,
                                                  self.positions_encoder, self.directions_encoder)

                optim.zero_grad()
                loss_coarse = self.loss_func(rgb, rgb_truth)
                loss_fine = self.loss_func(rgb_fine, rgb_truth)
                loss = loss_coarse + loss_fine

                loss.backward()
                optim.step()

                loss_item = loss.item()
                if i % log_nth == log_nth - 1:
                    print('[Epoch %d, Iteration %5d/%5d] TRAIN loss: %.7f' %
                          (epoch + 1, i + 1, iter_per_epoch, loss_item))
                    model_coarse.eval()
                    model_fine.eval()
                    val_loss = 0
                    for j, data in enumerate(val_loader):
                        ray_samples, ray_translation, ray_direction, z_vals, rgb_truth = data
                        ray_samples = ray_samples.to(device)  # [batchsize, number_coarse_samples, 3]
                        ray_translation = ray_translation.to(device)  # [batchsize, number_coarse_samples, 3]
                        ray_direction = ray_direction.to(device)  # [batchsize, number_coarse_samples, 3]
                        z_vals = z_vals.to(device)  # [batchsize, number_coarse_samples]
                        rgb_truth = rgb_truth.to(device)  # [batchsize, 3]

                        rgb, rgb_fine = run_nerf_pipeline(ray_samples, ray_translation, ray_direction, z_vals,
                                                          model_coarse, model_fine, self.sigma_noise_std,
                                                          self.number_fine_samples, self.white_background,
                                                          self.positions_encoder, self.directions_encoder)

                        loss_coarse = self.loss_func(rgb, rgb_truth)
                        loss_fine = self.loss_func(rgb_fine, rgb_truth)
                        loss = loss_coarse + loss_fine
                        val_loss += loss.item()
                    self.writer.add_scalars('Loss curve every nth iteration', {'train loss': loss_item,
                                                                               'val loss': val_loss / len(val_loader)},
                                            i // log_nth + epoch * (iter_per_epoch // log_nth))

                train_loss += loss_item
                self.train_loss_history_per_iter.append(loss_item)
            self.train_loss_history.append(train_loss / iter_per_epoch)
            print('[Epoch %d] Average loss of Epoch: %.7f' %
                  (epoch + 1, train_loss / iter_per_epoch))

            model_coarse.eval()
            model_fine.eval()
            val_loss = 0
            for i, data in enumerate(val_loader):
                ray_samples, ray_translation, ray_direction, z_vals, rgb_truth = data
                ray_samples = ray_samples.to(device)  # [batchsize, number_coarse_samples, 3]
                ray_translation = ray_translation.to(device)  # [batchsize, number_coarse_samples, 3]
                ray_direction = ray_direction.to(device)  # [batchsize, number_coarse_samples, 3]
                z_vals = z_vals.to(device)  # [batchsize, number_coarse_samples]
                rgb_truth = rgb_truth.to(device)  # [batchsize, 3]

                rgb, rgb_fine = run_nerf_pipeline(ray_samples, ray_translation, ray_direction, z_vals,
                                                  model_coarse, model_fine, self.sigma_noise_std,
                                                  self.number_fine_samples, self.white_background,
                                                  self.positions_encoder, self.directions_encoder)

                loss_coarse = self.loss_func(rgb, rgb_truth)
                loss_fine = self.loss_func(rgb_fine, rgb_truth)
                loss = loss_coarse + loss_fine
                val_loss += loss.item()

            print('[Epoch %d] VAL loss: %.7f' % (epoch + 1, val_loss))
            self.val_loss_history.append(val_loss)
            self.writer.add_scalars('Loss Curve', {'train loss': train_loss / iter_per_epoch,
                                                   'val loss': val_loss / len(val_loader)}, epoch)
        print('FINISH.')
