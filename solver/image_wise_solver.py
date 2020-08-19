import pyrender
import torch
import numpy as np
import trimesh

from torch.nn import functional as F

from camera import get_sphere_pose
from datasets.sub_dataset import SubDataset
from solver.nerf_solver import NerfSolver
from utils import PositionalEncoder, tensorboard_rerenders, vedo_data, raw2outputs
from torch.utils.data import DataLoader


class ImageWiseSolver(NerfSolver):
    '''
    Solver for a dataset of images and the corresponding rays such that an the warp of each ray can be calculated and the
    batching happens in an image.
    '''

    def __init__(self, model_coarse, model_fine, smpl_estimator, smpl_model, positions_encoder: PositionalEncoder,
                 directions_encoder: PositionalEncoder, args,
                 optim=torch.optim.Adam, loss_func=torch.nn.MSELoss()):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.smpl_estimator = smpl_estimator.to(self.device)
        self.smpl_model = smpl_model.to(self.device)
        self.position_encoder = positions_encoder
        self.direction_encoder = directions_encoder
        self.canonical_pose = torch.zeros([1, 69], device=self.device)
        super(ImageWiseSolver, self).__init__(model_coarse, model_fine, positions_encoder, directions_encoder, args,
                                              optim, loss_func)
        print('estimator params', list(self.smpl_estimator.parameters()))
        self.optim = optim([
            {'params': model_coarse.parameters()},
            {'params': self.smpl_estimator.parameters(), 'lr': args.lrate_pose}
        ],
            **self.optim_args_merged)

    def init_pipeline(self):
        return None

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

        print('START TRAIN.')

        for epoch in range(args.num_epochs):  # loop over the dataset multiple times
            self.model_coarse.train()
            self.model_fine.train()
            self.smpl_estimator.train()
            train_loss = 0
            pose_loss = 0
            for i, image_batch in enumerate(train_loader):
                for j, element in enumerate(image_batch):
                    image_batch[j] = element[0].to(self.device)
                ray_samples, samples_translations, samples_directions, z_vals, rgb = image_batch

                sub_dataset = SubDataset(ray_samples, samples_translations, samples_directions, rgb)
                dataloader = DataLoader(sub_dataset, args.batchsize, shuffle=True, num_workers=0)
                iter_per_image = len(dataloader)

                goal_pose, betas = self.smpl_estimator(1)
                # goal_pose[0, 41].register_hook(lambda x: print_max('goal_pose_grads', x))
                # self.smpl_estimator.arm_angle_r.register_hook(lambda x: print_max('goal_pose_grads', x))

                canonical_model = self.smpl_model(betas=betas, return_verts=True,
                                                  body_pose=self.canonical_pose)  # [number_vertices, 3]
                goal_models = self.smpl_model(betas=betas, return_verts=True, body_pose=goal_pose)

                goal_vertices = goal_models.vertices  # [1, number_vertices, 3]
                warp = canonical_model.vertices - goal_vertices  # [1, number_vertices, 3]
                warp = warp.expand(args.batchsize, -1, -1)
                for j, ray_batch in enumerate(dataloader):
                    for c, element in enumerate(ray_batch):
                        ray_batch[c] = element.to(self.device)
                    ray_samples, rays_translation, rays_direction, rgb_truth = ray_batch

                    distances = ray_samples[:, :, None, :] - goal_vertices[:, None, :, :].expand(
                        (-1, ray_samples.shape[1], -1, -1))  # [batchsize, number_samples, number_vertices, 3]
                    distances = torch.norm(distances, dim=-1)  # [batchsize, number_samples, number_vertices]
                    attentions = distances - self.args.warp_radius  # [batchsize, number_samples, number_vertices]
                    attentions = F.relu(-attentions)

                    # attentions = torch.softmax(self.args.warp_temperature * attentions, dim=-1)
                    attentions = attentions / (attentions.sum(-1, keepdims=True) + 1e-5)

                    warps = warp[:, None, :, :] * attentions[:, :, :,
                                                  None]  # [batchsize, number_samples, number_vertices, 3]
                    warps = warps.sum(dim=-2)  # [batchsize, number_samples, 3]
                    warped_samples = ray_samples + warps

                    samples_encoding = self.position_encoder.encode(warped_samples)

                    coarse_samples_directions = warped_samples - rays_translation[:, None,
                                                                 :]  # [batchsize, number_coarse_samples, 3]
                    samples_directions_norm = coarse_samples_directions / torch.norm(coarse_samples_directions, dim=-1,
                                                                                     keepdim=True)
                    directions_encoding = self.direction_encoder.encode(samples_directions_norm)
                    # flatten the encodings from [batchsize, number_coarse_samples, encoding_size] to [batchsize * number_coarse_samples, encoding_size] and concatenate
                    inputs = torch.cat([samples_encoding.view(-1, samples_encoding.shape[-1]),
                                        directions_encoding.view(-1, directions_encoding.shape[-1])], -1)
                    raw_outputs = self.model_coarse(inputs)  # [batchsize * number_coarse_samples, 4]
                    raw_outputs = raw_outputs.view(samples_encoding.shape[0], samples_encoding.shape[1],
                                                   raw_outputs.shape[-1])  # [batchsize, number_coarse_samples, 4]
                    rgb, weights, densities = raw2outputs(raw_outputs, z_vals, coarse_samples_directions, self.args)

                    self.optim.zero_grad()
                    loss = self.loss_func(rgb, rgb_truth)

                    loss.backward(retain_graph=True)
                    self.optim.step()

                    loss_item = loss.item()
                    left_arm_loss = (self.smpl_estimator.arm_angle_l[0] -
                                     self.smpl_estimator.ground_truth_pose[0, 38]) ** 2
                    right_arm_loss = (self.smpl_estimator.arm_angle_r[0] -
                                      self.smpl_estimator.ground_truth_pose[0, 41]) ** 2
                    pose_loss_item = (left_arm_loss + right_arm_loss).item()

                    if j % args.log_iterations == args.log_iterations - 1:
                        print('[Epoch %d, Iteration %5d/%5d] TRAIN loss: %.7f Pose Loss: %.7f' %
                              (epoch + 1, j + 1, iter_per_image, loss_item, pose_loss_item))

                    pose_loss += pose_loss_item
                    train_loss += loss_item
            print('[Epoch %d] Average loss of Epoch: %.7f Pose Loss: %.7f' %
                  (epoch + 1, train_loss / iter_per_image * len(train_loader),
                   pose_loss / iter_per_image * len(train_loader)))

            self.model_coarse.eval()
            self.model_fine.eval()
            self.smpl_estimator.eval()
            val_loss = 0
            rerender_images = []
            ground_truth_images = []
            samples = []
            ray_warp_magnitudes = []
            densities_list = []
            for i, image_batch in enumerate(val_loader):
                for j, element in enumerate(image_batch):
                    image_batch[j] = element[0].to(self.device)
                ray_samples, samples_translations, samples_directions, z_vals, rgb = image_batch

                sub_dataset = SubDataset(ray_samples, samples_translations, samples_directions, rgb)
                dataloader = DataLoader(sub_dataset, args.batchsize, shuffle=False, num_workers=0)
                iter_per_image_val = len(dataloader)
                goal_pose, betas = self.smpl_estimator(1)

                canonical_model = self.smpl_model(betas=betas, return_verts=True,
                                                  body_pose=self.canonical_pose)  # [number_vertices, 3]
                goal_models = self.smpl_model(betas=betas, return_verts=True, body_pose=goal_pose)

                goal_vertices = goal_models.vertices  # [1, number_vertices, 3]
                warp = canonical_model.vertices - goal_vertices  # [1, number_vertices, 3]
                warp = warp.expand(args.batchsize, -1, -1)  # [batchsize, number_vertices, 3]
                image_warps = []
                image_densities = []
                image_samples = []
                for j, ray_batch in enumerate(dataloader):
                    for j, element in enumerate(ray_batch):
                        ray_batch[j] = element.to(self.device)
                    ray_samples, rays_translation, rays_direction, rgb_truth = ray_batch

                    distances = ray_samples[:, :, None, :] - goal_vertices[:, None, :, :].expand(
                        (-1, ray_samples.shape[1], -1, -1))  # [batchsize, number_samples, number_vertices, 3]
                    distances = torch.norm(distances, dim=-1)  # [batchsize, number_samples, number_vertices]
                    attentions = distances - self.args.warp_radius  # [batchsize, number_samples, number_vertices]
                    attentions = F.relu(-attentions)

                    # attentions = torch.softmax(self.args.warp_temperature * attentions, dim=-1)
                    attentions = attentions / (attentions.sum(-1, keepdims=True) + 1e-5)

                    warps = warp[:, None, :, :] * attentions[:, :, :,
                                                  None]  # [batchsize, number_samples, number_vertices, 3]
                    warps = warps.sum(dim=-2)  # [batchsize, number_samples, 3]
                    warped_samples = ray_samples + warps

                    samples_encoding = self.position_encoder.encode(warped_samples)

                    coarse_samples_directions = warped_samples - rays_translation[:, None,
                                                                 :]  # [batchsize, number_coarse_samples, 3]
                    samples_directions_norm = coarse_samples_directions / torch.norm(coarse_samples_directions, dim=-1,
                                                                                     keepdim=True)
                    directions_encoding = self.direction_encoder.encode(samples_directions_norm)
                    # flatten the encodings from [batchsize, number_coarse_samples, encoding_size] to [batchsize * number_coarse_samples, encoding_size] and concatenate
                    inputs = torch.cat([samples_encoding.view(-1, samples_encoding.shape[-1]),
                                        directions_encoding.view(-1, directions_encoding.shape[-1])], -1)
                    raw_outputs = self.model_coarse(inputs)  # [batchsize * number_coarse_samples, 4]
                    raw_outputs = raw_outputs.view(samples_encoding.shape[0], samples_encoding.shape[1],
                                                   raw_outputs.shape[-1])  # [batchsize, number_coarse_samples, 4]
                    rgb, weights, densities = raw2outputs(raw_outputs, z_vals, coarse_samples_directions, self.args)

                    loss = self.loss_func(rgb, rgb_truth)

                    val_loss += loss.item()

                    ground_truth_images.append(rgb_truth.detach().cpu().numpy())
                    rerender_images.append(rgb.detach().cpu().numpy())
                    samples.append(ray_samples.detach().cpu().numpy())
                    image_samples.append(ray_samples.detach().cpu().numpy())
                    image_warps.append(warps.detach().cpu().numpy())
                    densities_list.append(densities.detach().cpu().numpy())
                    image_densities.append(densities.detach().cpu().numpy())

                    warp_magnitude = np.linalg.norm(warps.detach().cpu(), axis=-1)  # [batchsize, number_samples]
                    ray_warp_magnitudes.append(warp_magnitude.mean(axis=1))  # mean over the samples => [batchsize]

                vedo_data(self.writer, np.concatenate(image_densities).reshape(-1),
                          np.concatenate(image_samples).reshape(-1, 3),
                          image_warps=np.concatenate(image_warps).reshape(-1, 3), epoch=epoch + 1,
                          image_idx=i)
            if len(val_loader) != 0:
                rerender_images = np.concatenate(rerender_images, 0).reshape((-1, h, w, 3))
                ground_truth_images = np.concatenate(ground_truth_images).reshape((-1, h, w, 3))
                ray_warp_magnitudes = np.concatenate(ray_warp_magnitudes).reshape((-1, h, w))

            tensorboard_rerenders(self.writer, args.number_validation_images, rerender_images, ground_truth_images,
                                  step=epoch + 1, ray_warps=ray_warp_magnitudes)

            goal_model = self.smpl_model(betas=betas, return_verts=True, body_pose=goal_pose)

            mesh = trimesh.Trimesh(goal_model.vertices.detach().cpu().numpy()[0], self.smpl_model.faces, process=False)
            mesh = pyrender.Mesh.from_trimesh(mesh)

            scene = pyrender.Scene()
            scene.add(mesh, pose=get_sphere_pose(0, 0, 0))
            camera = pyrender.PerspectiveCamera(yfov=np.pi / 3, aspectRatio=1.0)
            scene.add(camera, pose=get_sphere_pose(0, 0, 2.4))
            light = pyrender.SpotLight(color=np.ones(3), intensity=200.0,
                                       innerConeAngle=np.pi / 16.0,
                                       outerConeAngle=np.pi / 6.0)
            scene.add(light, pose=get_sphere_pose(0, 0, 2.4))
            r = pyrender.OffscreenRenderer(128, 128)
            img, depth = r.render(scene)
            img = img.copy()

            self.writer.add_image(str(epoch + 1) + ' Smpl', torch.from_numpy(img).permute(2,0,1), epoch + 1)

            print('[Epoch %d] VAL loss: %.7f' % (
                epoch + 1,
                val_loss / (len(val_loader) * iter_per_image_val or not len(val_loader) * iter_per_image_val)))
            self.writer.add_scalars('Loss Curve', {'train loss': train_loss / iter_per_image * len(train_loader),
                                                   'val loss': val_loss / (
                                                           len(val_loader) * iter_per_image_val or not len(
                                                       val_loader) * iter_per_image_val)},
                                    epoch + 1)
            self.writer.add_scalar('Pose difference', pose_loss / iter_per_image * len(train_loader), epoch + 1)
        print('FINISH.')
