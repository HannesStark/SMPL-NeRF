from collections import defaultdict

import torch

from models.nerf_pipeline import NerfPipeline
from utils import PositionalEncoder, raw2outputs, modified_softmax, print_max, print_number_nans
from torch.nn import functional as F


class DynamicPipeline(NerfPipeline):

    def __init__(self, model_coarse, model_fine, smpl_estimator, smpl_model,
                 args, position_encoder: PositionalEncoder,
                 direction_encoder: PositionalEncoder):
        super(DynamicPipeline, self).__init__(model_coarse, model_fine, args, position_encoder, direction_encoder)
        self.smpl_estimator = smpl_estimator
        self.smpl_model = smpl_model
        self.global_orient = torch.zeros([1, 3], device=self.device)
        self.canonical_pose = torch.zeros([1, 69], device=self.device)
        self.args = args
        self.pre_attention_warps = None

    def forward(self, data):
        """
            Volumetric rendering with NeRF.

            Returns
            -------
            rgb : torch.Tensor ([batch_size, 3])
                Estimated RGB color with coarse net.
            rgb_fine : torch.Tensor ([batch_size, 3])
                Estimated RGB color with fine net.
            """
        ray_samples, ray_translation, ray_direction, z_vals, images, rb_truth = data

        goal_poses, expressions, betas = self.smpl_estimator(images)
        #print('betas ', self.smpl_estimator.betas)
        #print('expression ', self.smpl_estimator.expression)
        #print('goal_poses', self.smpl_estimator.goal_poses)

        # right now expanding and using self.global_orient_instead of setting the correct batchisze for the smpl model since the same smpl model is used in train and validation with different batch sizes (the batchisze of the smpl_model is 1 by default)
        global_orient = self.global_orient.expand(len(ray_samples), -1)
        canonical_pose = self.canonical_pose.expand(len(ray_samples), -1)

        canonical_model = self.smpl_model(betas=betas, expression=expressions, return_verts=True,
                                          body_pose=canonical_pose,
                                          global_orient=global_orient)  # [number_vertices, 3]
        goal_models = self.smpl_model(betas=betas, expression=expressions, return_verts=True, body_pose=goal_poses,
                                      global_orient=global_orient)
        goal_vertices = goal_models.vertices  # [batchsize, number_vertices, 3]
        warps = canonical_model.vertices - goal_vertices  # [batchsize, number_vertices, 3]

        distances = ray_samples[:, :, None, :] - goal_vertices[:, None, :, :].expand(
            (-1, ray_samples.shape[1], -1, -1))  # [batchsize, number_samples, number_vertices, 3]
        distances = torch.norm(distances, dim=-1)  # [batchsize, number_samples, number_vertices]
        attentions_1 = distances - self.args.warp_radius  # [batchsize, number_samples, number_vertices]
        attentions_2 = F.relu(-attentions_1)
        #print('iter')
        #attentions_2.register_hook(lambda x: print_number_nans('pre', x))
        #attentions_2.register_hook(lambda x: print_max('pre',x))

        attentions_3 = modified_softmax(self.args.warp_temperature * attentions_2)
        #attentions_3.register_hook(lambda x: print_max('post',x))
        warps = warps[:, None, :, :] * attentions_3[:, :, :, None]  # [batchsize, number_samples, number_vertices, 3]
        warps = warps.sum(dim=-2)  # [batchsize, number_samples, 3]
        warped_samples = ray_samples + warps

        samples_encoding = self.position_encoder.encode(warped_samples)

        coarse_samples_directions = warped_samples - ray_translation[:, None,
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

        return rgb, rgb, warps, ray_samples, warped_samples, densities
