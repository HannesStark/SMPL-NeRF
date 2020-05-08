import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
from utils import run_nerf_pipeline, PositionalEncoder, run_smpl_nerf_pipeline


class SmplNerfSolver():
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0}

    def __init__(self, positions_encoder: PositionalEncoder, directions_encoder: PositionalEncoder,
                 optim=torch.optim.Adam, optim_args={},
                 loss_func=torch.nn.MSELoss(), sigma_noise_std: float = 1,
                 white_background: bool = False, number_fine_samples: int = 128):
        """
        Parameters
        ----------
        positions_encoder : PositionalEncoder
        directions_encoder : PositionalEncoder
        optim : torch.optim, optional
            used optimizer. The default is torch.optim.Adam.
        optim_args : dict, optional
            settings for optimizer. The default is {}.
        loss_func : torch.nn, optional
            loss for training. The default is torch.nn.MSELoss().
        sigma_noise_std : float, optional
            regularization. The default is 1.
        white_background : bool, optional
            The default is False.
        number_fine_samples : int, optional
            The default is 128.
        """
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

    def train(self, model_coarse, model_fine, model_smpl, train_loader, val_loader,
              h: int, w: int, num_epochs: int = 10, log_nth: int = 0,
              number_validation_images: int = 0, early_validation: bool = False):
        """
        Train coarse and fine model on training data and run validation

        Parameters
        ----------
        model_coarse : coarse model object.
        model_fine : fine model object.
        train_loader : training data loader object.
        val_loader : validation data loader object.
        h : int
            height of images.
        w : int
            width of images.
        num_epochs : int, optional
            total number of training epochs. The default is 10.
        log_nth : int, optional
            log train loss every nth iteration. The default is 0.
        number_validation_images : int, optional
            plot n validation images with groundtruth in tensorboard.
            The default is 0.
        early_validation : bool, optional
            perform validation step every nth iteration. The default is False.

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
                ray_samples, ray_translation, ray_direction, z_vals, rgb_truth, dependency_rays, goal_pose = data
                ray_samples = ray_samples.to(device)  # [batchsize, number_coarse_samples, 3]
                ray_translation = ray_translation.to(device)  # [batchsize, 3]
                ray_direction = ray_direction.to(device)  # [batchsize, 3]
                z_vals = z_vals.to(device)  # [batchsize, number_coarse_samples]
                rgb_truth = rgb_truth.to(device)  # [batchsize, 3]

                rgb, rgb_fine = run_smpl_nerf_pipeline(ray_samples, ray_translation, ray_direction, z_vals,
                                                       dependency_rays, goal_pose,
                                                       model_coarse, model_fine, model_smpl, self.sigma_noise_std,
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
                    if early_validation:
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

                            rgb, rgb_fine = run_smpl_nerf_pipeline(ray_samples, ray_translation, ray_direction, z_vals,
                                                                   dependency_rays, goal_pose,
                                                                   model_coarse, model_fine, model_smpl,
                                                                   self.sigma_noise_std,
                                                                   self.number_fine_samples, self.white_background,
                                                                   self.positions_encoder, self.directions_encoder)

                            loss_coarse = self.loss_func(rgb, rgb_truth)
                            loss_fine = self.loss_func(rgb_fine, rgb_truth)
                            loss = loss_coarse + loss_fine
                            val_loss += loss.item()
                        self.writer.add_scalars('Loss curve every nth iteration', {'train loss': loss_item,
                                                                                   'val loss': val_loss / len(
                                                                                       val_loader)},
                                                i // log_nth + epoch * (iter_per_epoch // log_nth))

                train_loss += loss_item
                self.train_loss_history_per_iter.append(loss_item)
            self.train_loss_history.append(train_loss / iter_per_epoch)
            print('[Epoch %d] Average loss of Epoch: %.7f' %
                  (epoch + 1, train_loss / iter_per_epoch))

            model_coarse.eval()
            model_fine.eval()
            val_loss = 0
            rerender_images = []
            ground_truth_images = []
            for i, data in enumerate(val_loader):
                ray_samples, ray_translation, ray_direction, z_vals, rgb_truth = data
                ground_truth_images.append(rgb_truth)
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
                rerender_images.append(rgb_fine.detach().cpu().numpy())

            rerender_images = np.concatenate(rerender_images, 0).reshape((-1, h, w, 3))
            ground_truth_images = np.concatenate(ground_truth_images).reshape((-1, h, w, 3))
            if number_validation_images > rerender_images.shape[0]:
                print('there are only ', rerender_images.shape[0],
                      ' in the validation directory which is less than the specified number_validation_images: ',
                      number_validation_images, ' So instead ', rerender_images.shape[0],
                      ' images are sent to tensorboard')
                number_validation_images = rerender_images.shape[0]
            else:
                rerender_images = rerender_images[:number_validation_images]

            fig, axarr = plt.subplots(number_validation_images, 2, sharex=True, sharey=True)
            if len(axarr.shape) == 1:
                axarr = axarr[None, :]
            for i in range(number_validation_images):
                # strange indices after image because matplotlib wants bgr instead of rgb
                axarr[i, 0].imshow(ground_truth_images[i][:, :, ::-1])
                axarr[i, 0].axis('off')
                axarr[i, 1].imshow(rerender_images[i][:, :, ::-1])
                axarr[i, 1].axis('off')
            axarr[0, 0].set_title('Ground Truth')
            axarr[0, 1].set_title('Rerender')
            fig.set_dpi(300)
            self.writer.add_figure(str(epoch) + ' validation images', fig, epoch)

            print('[Epoch %d] VAL loss: %.7f' % (epoch + 1, val_loss / len(val_loader)))
            self.val_loss_history.append(val_loss)
            self.writer.add_scalars('Loss Curve', {'train loss': train_loss / iter_per_epoch,
                                                   'val loss': val_loss / len(val_loader)}, epoch)
        print('FINISH.')
