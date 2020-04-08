from pystoi import stoi
from pesq import pesq
import librosa
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision
import librosa.display
from torch.utils.tensorboard import SummaryWriter

from utils import positional_encoding, raw2outputs, fine_sampling


class Solver():
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0}

    def __init__(self, optim=torch.optim.Adam, optim_args={}, loss_func=torch.nn.MSELoss(), sigma_noise_std=1,
                 white_background=False, number_fine_samples=128):
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func
        self.sigma_noise_std = sigma_noise_std
        self.white_background = white_background
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
        if torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        model_coarse.to(device)
        model_fine.to(device)

        print('START TRAIN.')

        for epoch in range(num_epochs):  # loop over the dataset multiple times
            model_coarse.train()
            model_fine.train()
            train_loss = 0
            for i, data in enumerate(train_loader):
                ray_samples, samples_translations, samples_directions, z_vals, rgb_truth = data
                ray_samples = ray_samples.to(device)  # [batchsize, number_coarse_samples, 3]
                samples_translations = samples_translations.to(device)  # [batchsize, number_coarse_samples, 3]
                samples_directions = samples_directions.to(device)  # [batchsize, number_coarse_samples, 3]
                z_vals = z_vals.to(device)  # [batchsize, number_coarse_samples]
                rgb_truth = rgb_truth.to(device)  # [batchsize, 3]

                # get values for coarse network and run them through the coarse network
                samples_encoding = positional_encoding(ray_samples, 10, True)
                samples_directions_norm = samples_directions / torch.norm(samples_directions, dim=-1, keepdim=True)
                directions_encoding = positional_encoding(samples_directions_norm, 4, False)
                # flatten the encodings from [batchsize, number_coarse_samples, encoding_size] to [batchsize * number_coarse_samples, encoding_size] and concatenate
                inputs = torch.cat([samples_encoding.view(-1, samples_encoding.shape[-1]),
                                    directions_encoding.view(-1, directions_encoding.shape[-1])], -1)
                optim.zero_grad()
                raw_outputs = model_coarse(inputs)  # [batchsize * number_coarse_samples, 4]
                raw_outputs = raw_outputs.view(samples_encoding.shape[0], samples_encoding.shape[1],
                                               raw_outputs.shape[-1])  # [batchsize, number_coarse_samples, 4]
                rgb, weights = raw2outputs(raw_outputs, z_vals, samples_directions, self.sigma_noise_std,
                                           self.white_background)

                # get values for the fine network and run them through the fine network
                ray_samples_fine = fine_sampling(samples_translations, samples_directions, z_vals, weights,
                                                 self.number_fine_samples)  # [batchsize, number_coarse_samples + number_fine_samples, 3]
                samples_encoding_fine = positional_encoding(ray_samples_fine, 10, True)
                # expand directions and translations to the number of coarse samples + fine_samples
                directions_encoding_fine = directions_encoding[..., :1, :].expand(directions_encoding.shape[0],
                                                                                  ray_samples_fine.shape[1],
                                                                                  directions_encoding.shape[-1])
                inputs_fine = torch.cat([samples_encoding_fine.view(-1, samples_encoding_fine.shape[-1]),
                                         directions_encoding_fine.view(-1, samples_encoding_fine.shape[-1])], -1)
                raw_outputs = model_coarse(
                    inputs_fine)  # [batchsize * (number_coarse_samples + number_fine_samples), 4]
                raw_outputs_fine = raw_outputs.view(samples_encoding_fine.shape[0], samples_encoding_fine.shape[1],
                                                    raw_outputs_fine.shape[
                                                        -1])  # [batchsize, number_coarse_samples + number_fine_samples, 4]
                # expand directions and translations to the number of coarse samples + fine_samples
                samples_directions_fine = samples_directions[..., :1, :].expand(samples_directions.shape[0],
                                                                                ray_samples_fine.shape[1],
                                                                                samples_directions.shape[-1])
                rgb_fine, _ = raw2outputs(raw_outputs_fine, z_vals, samples_directions_fine, self.sigma_noise_std,
                                          self.white_background)

                loss = self.loss_func(outputs, labels)
                loss.backward()
                optim.step()

                loss_item = loss.item()
                if i % log_nth == log_nth - 1:
                    print('[Epoch %d, Iteration %5d/%5d] TRAIN loss: %.7f' %
                          (epoch + 1, i + 1, iter_per_epoch, loss_item))

                train_loss += loss_item
                self.train_loss_history_per_iter.append(loss_item)
            self.train_loss_history.append(train_loss / iter_per_epoch)
            print('[Epoch %d] Average loss of Epoch: %.7f' %
                  (epoch + 1, train_loss / iter_per_epoch))

            model.eval()
            val_loss = 0
            for i, data in enumerate(val_loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model.forward(inputs)
                loss = self.loss_func(outputs, labels)
                val_loss += loss.item()

                inputs = inputs.cpu().detach().numpy()
                outputs = outputs.cpu().detach().numpy()
                labels = labels.cpu().detach().numpy()

            print('[Epoch %d] VAL loss: %.7f' % (epoch + 1, val_loss))
            self.val_loss_history.append(val_loss)
            self.writer.add_scalars('Loss Curve', {'train loss': train_loss / iter_per_epoch,
                                                   'val loss': val_loss / iter_per_epoch}, epoch)
        print('FINISH.')
