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

        optim = self.optim(list(model_coarse.parameters()) + list(model_fine.parameters()))
        self._reset_histories()
        iter_per_epoch = len(train_loader)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
                directions_encoding = positional_encoding(samples_directions, 4, False)
                # flatten the encodings from [batchsize, number_coarse_samples, encoding_size] to [batchsize * number_coarse_samples, encoding_size] and concatenate
                inputs = torch.cat([samples_encoding.view(-1, samples_encoding.shape[-1]),
                                    directions_encoding.view(-1, directions_encoding.shape[-1])], -1)
                optim.zero_grad()
                raw_outputs = model_coarse(inputs)  # [batchsize * number_coarse_samples, 4]
                raw_outputs = raw_outputs.view(samples_encoding.shape[0], samples_encoding.shape[1],
                                               raw_outputs.shape[-1])  # [batchsize, number_coarse_samples, 4]
                rgb, weights = raw2outputs(raw_outputs, z_vals, directions_encoding, self.sigma_noise_std,
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
                raw_outputs = model_coarse(inputs_fine)  # [batchsize * (number_coarse_samples + number_fine_samples), 4]
                raw_outputs_fine = raw_outputs.view(samples_encoding_fine.shape[0], samples_encoding_fine.shape[1],
                                                    raw_outputs_fine.shape[-1])  # [batchsize, number_coarse_samples + number_fine_samples, 4]
                rgb_fine, _ = raw2outputs(raw_outputs_fine, z_vals, samples_directions, self.sigma_noise_std,
                                          self.white_background)

                loss = self.loss_func(outputs, labels)
                loss.backward()
                optim.step()

                loss_item = loss.item()
                if i % log_nth == log_nth - 1:
                    print("difference between inputs and predictions: " + str(self.loss_func(inputs, outputs).item()))
                    print("difference between inputs and labels:      " + str(self.loss_func(inputs, labels).item()))
                    print('[Epoch %d, Iteration %5d/%5d] TRAIN loss: %.7f' %
                          (epoch + 1, i + 1, iter_per_epoch, loss_item))

                train_loss += loss_item
                self.train_loss_history_per_iter.append(loss_item)
            self.train_loss_history.append(train_loss / iter_per_epoch)
            print('[Epoch %d] Average loss of Epoch: %.7f' %
                  (epoch + 1, train_loss / iter_per_epoch))

            model.eval()
            val_loss = 0
            stoi_clean_noisy = 0
            stoi_clean_output = 0
            pesq_clean_noisy = 0
            pesq_clean_output = 0
            for i, data in enumerate(val_loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model.forward(inputs)
                loss = self.loss_func(outputs, labels)
                val_loss += loss.item()

                inputs = inputs.cpu().detach().numpy()
                outputs = outputs.cpu().detach().numpy()
                labels = labels.cpu().detach().numpy()
                for j in range(len(inputs)):
                    stoi_clean_noisy += stoi(labels[j][0], inputs[j][0], fs_sig=16000)
                    stoi_clean_output += stoi(labels[j][0], outputs[j][0], fs_sig=16000)
                    pesq_clean_noisy += pesq(16000, labels[j][0], inputs[j][0], mode='wb')
                    pesq_clean_output += pesq(16000, labels[j][0], outputs[j][0], mode='wb')

                if i + 1 == len(val_loader):
                    self.writer.add_scalars('STOI', {'enhanced': stoi_clean_output / (iter_per_epoch * len(inputs)),
                                                     'mixture': stoi_clean_noisy / (iter_per_epoch * len(inputs))},
                                            epoch)
                    self.writer.add_scalars('PESQ', {'enhanced': pesq_clean_output / (iter_per_epoch * len(inputs)),
                                                     'mixture': pesq_clean_noisy / (iter_per_epoch * len(inputs))},
                                            epoch)

                    # draw plots for the first tensorboard_plots plots in the batch
                    for j in range(min(len(inputs), tensorboard_plots)):
                        # waveform
                        fig1, axes1 = plt.subplots(3, 1, sharex='all')
                        axes1[0].set_ylabel('mixed')
                        axes1[1].set_ylabel('cleaned')
                        axes1[2].set_ylabel('truth')
                        for k, audio in enumerate([inputs, outputs, labels]):
                            librosa.display.waveplot(audio[j][0], sr=1600, ax=axes1[k], x_axis=None)
                            axes1[k].set_ylim((-1, 1))
                        self.writer.add_figure(str(j) + ' of batch', fig1, epoch)

                        # spectogram
                        inputs_mag, _ = librosa.magphase(
                            librosa.stft(inputs[j][0], n_fft=320, hop_length=160, win_length=320))
                        outputs_mag, _ = librosa.magphase(
                            librosa.stft(outputs[j][0], n_fft=320, hop_length=160, win_length=320))
                        labels_mag, _ = librosa.magphase(
                            librosa.stft(labels[j][0], n_fft=320, hop_length=160, win_length=320))

                        fig2, axes2 = plt.subplots(3, 1, figsize=(6, 6))
                        axes2[0].set_ylabel('mixed')
                        axes2[1].set_ylabel('cleaned')
                        axes2[2].set_ylabel('truth')
                        for k, mag in enumerate([inputs_mag, outputs_mag, labels_mag]):
                            librosa.display.specshow(librosa.amplitude_to_db(mag), cmap="magma", y_axis="linear",
                                                     ax=axes2[k])
                        self.writer.add_figure(str(j) + ' spectogram of batch', fig2, epoch)

            print('[Epoch %d] VAL loss: %.7f' % (epoch + 1, val_loss))
            self.val_loss_history.append(val_loss)
            self.writer.add_scalars('Loss Curve', {'train loss': train_loss / iter_per_epoch,
                                                   'val loss': val_loss / iter_per_epoch}, epoch)
        print('FINISH.')
