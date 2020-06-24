import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from models.nerf_pipeline import NerfPipeline
from utils import PositionalEncoder, tensorboard_rerenders, pyrender_data


class NerfSolver():
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0}

    def __init__(self, model_coarse, model_fine, positions_encoder: PositionalEncoder,
                 directions_encoder: PositionalEncoder,
                 args, optim=torch.optim.Adam, loss_func=torch.nn.MSELoss()):
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
        """

        self.optim_args_merged = self.default_adam_args.copy()
        self.optim_args_merged.update({"lr": args.lrate, "weight_decay": args.weight_decay})
        self.optim = optim(list(model_coarse.parameters()) + list(model_fine.parameters()), **self.optim_args_merged)
        self.loss_func = loss_func
        self.positions_encoder = positions_encoder
        self.directions_encoder = directions_encoder
        self.writer = SummaryWriter()
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        self.model_coarse = model_coarse.to(self.device)
        self.model_fine = model_fine.to(self.device)
        self.pipeline = self.init_pipeline()

    def init_pipeline(self):
        return NerfPipeline(self.model_coarse, self.model_fine, self.args, self.positions_encoder,
                            self.directions_encoder)

    def nerf_loss(self, rgb, rgb_fine, rgb_truth):
        loss_coarse = self.loss_func(rgb, rgb_truth)
        loss_fine = self.loss_func(rgb_fine, rgb_truth)
        loss = loss_coarse + loss_fine
        return loss

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
            for i, data in enumerate(train_loader):
                for j, element in enumerate(data):
                    data[j] = element.to(self.device)
                rgb_truth = data[-1]

                rgb, rgb_fine, ray_samples, densities = self.pipeline(data)

                self.optim.zero_grad()

                loss = self.nerf_loss(rgb, rgb_fine, rgb_truth)
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

                            rgb, rgb_fine, _, _ = self.pipeline(data)

                            loss = self.nerf_loss(rgb, rgb_fine, rgb_truth)
                            val_loss += loss.item()
                        self.writer.add_scalars('Loss curve every nth iteration', {'train loss': loss_item,
                                                                                   'val loss': val_loss / len(
                                                                                       val_loader)},
                                                i // args.log_iterations + epoch * (
                                                        iter_per_epoch // args.log_iterations))
                train_loss += loss_item
            print('[Epoch %d] Average loss of Epoch: %.7f' %
                  (epoch + 1, train_loss / iter_per_epoch))

            self.model_coarse.eval()
            self.model_fine.eval()
            val_loss = 0
            rerender_images = []
            samples = []
            ground_truth_images = []
            densities_list = []
            for i, data in enumerate(val_loader):
                for j, element in enumerate(data):
                    data[j] = element.to(self.device)
                rgb_truth = data[-1]

                rgb, rgb_fine, ray_samples, densities = self.pipeline(data)

                loss = self.nerf_loss(rgb, rgb_fine, rgb_truth)
                val_loss += loss.item()

                ground_truth_images.append(rgb_truth.detach().cpu().numpy())
                rerender_images.append(rgb_fine.detach().cpu().numpy())
                samples.append(ray_samples.detach().cpu().numpy())
                densities_list.append(densities.detach().cpu().numpy())
            if len(val_loader) != 0:
                rerender_images = np.concatenate(rerender_images, 0).reshape((-1, h, w, 3))
                ground_truth_images = np.concatenate(ground_truth_images).reshape((-1, h, w, 3))

                samples = np.concatenate(samples)
                samples = samples.reshape(
                    (-1, h * w * samples.shape[-2], 3))  # [number_images, h*w*(n_fine_samples + n_coarse_samples), 3]

                densities_list = np.concatenate(densities_list)
                densities_list = densities_list.reshape(
                    (-1,
                     h * w * densities_list.shape[-1]))  # [number_images, h*w*(n_fine_samples + n_coarse_samples), 3]

            pyrender_data(self.writer, densities_list, samples, warps=None, step=epoch + 1)
            tensorboard_rerenders(self.writer, args.number_validation_images, rerender_images, ground_truth_images,
                                  step=epoch, warps=None)

            print('[Epoch %d] VAL loss: %.7f' % (epoch + 1, val_loss / (len(val_loader) or not len(val_loader))))
            self.writer.add_scalars('Loss Curve', {'train loss': train_loss / iter_per_epoch,
                                                   'val loss': val_loss / (len(val_loader) or not len(val_loader))},
                                    epoch)
        print('FINISH.')
