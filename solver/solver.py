from pystoi import stoi
from pesq import pesq
import librosa
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision
import librosa.display
from torch.utils.tensorboard import SummaryWriter

from utils import positional_encoding


class Solver():
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0}

    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 loss_func=torch.nn.MSELoss()):
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func
        self.writer = SummaryWriter()
        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_loss_history_per_iter = []
        self.val_loss_history = []

    def train(self, model, train_loader, val_loader, num_epochs=10, log_nth=0):
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
        optim = self.optim(model.parameters(), **self.optim_args)
        self._reset_histories()
        iter_per_epoch = len(train_loader)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        print('START TRAIN.')

        for epoch in range(num_epochs):  # loop over the dataset multiple times
            model.train()
            train_loss = 0
            for i, data in enumerate(train_loader):
                ray_samples, samples_translations, samples_directions, z_vals, rgbs_for_samples = data
                ray_samples = ray_samples.to(device)
                samples_translations = samples_translations.to(device)
                samples_directions = samples_directions.to(device)
                z_vals = z_vals.to(device)
                rgbs_for_samples = rgbs_for_samples.to(device)

                sample_encoding = positional_encoding(ray_samples, 10, True)
                direction_encoding = positional_encoding(samples_directions, 4, False)
                inputs = torch.cat([sample_encoding, direction_encoding], -1)
                optim.zero_grad()
                outputs = model(inputs)

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
