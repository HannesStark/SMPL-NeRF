import torch
import numpy as np

#from solver.nerf_solver import NerfSolver
from utils import tensorboard_rerenders, tensorboard_warps
from torch.utils.tensorboard import SummaryWriter

class SmplEstimatorSolver():
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0}

    def __init__(self, model_smpl_estimator, args, optim=torch.optim.Adam, loss_func=torch.nn.MSELoss()):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_smpl_estimator = model_smpl_estimator.to(self.device)
        self.optim_args_merged = self.default_adam_args.copy()
        self.optim_args_merged.update({"lr": args.lrate, "weight_decay": args.weight_decay})
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.args = args
        self.writer = SummaryWriter()
        self.optim = optim(list(model_smpl_estimator.parameters()),
                           **self.optim_args_merged)
        self.loss_func = loss_func




    def train(self, train_loader, val_loader):
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
                image_truth, goal_pose_truth = data
                goal_pose_truth = torch.stack([goal_pose_truth[:, 38], goal_pose_truth[:, 41]], axis=-1)
                goal_pose = self.model_smpl_estimator.forward(image_truth)
                self.optim.zero_grad()
                loss = self.loss_func(goal_pose, goal_pose_truth)
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

            for i, data in enumerate(val_loader):
                for j, element in enumerate(data):
                    data[j] = element.to(self.device)

                image_truth, goal_pose_truth = data
                with torch.no_grad():
                    goal_pose_truth = torch.stack([goal_pose_truth[:, 38], goal_pose_truth[:, 41]], axis=-1)
                    goal_pose = self.model_smpl_estimator.forward(image_truth)
                    loss = self.loss_func(goal_pose, goal_pose_truth)
                    val_loss += loss.item()

            print('[Epoch %d] VAL loss: %.7f' % (epoch + 1, val_loss / (len(val_loader) or not len(val_loader))))
            self.writer.add_scalars('Loss Curve', {'train loss': train_loss / iter_per_epoch,
                                                   'val loss': val_loss / (len(val_loader) or not len(val_loader))},
                                    epoch)
        print('FINISH.')