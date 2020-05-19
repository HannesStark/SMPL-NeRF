import torch

from models.nerf_pipeline import NerfPipeline
from utils import PositionalEncoder, raw2outputs, fine_sampling


class SmplNerfPipeline(NerfPipeline):

    def __init__(self, model_coarse, model_fine, model_warp_field, args, position_encoder: PositionalEncoder,
                 direction_encoder: PositionalEncoder, human_pose_encoder: PositionalEncoder):
        super(SmplNerfPipeline, self).__init__(model_coarse, model_fine, args, position_encoder, direction_encoder)
        self.human_pose_encoder = human_pose_encoder
        self.model_warp_field = model_warp_field

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
        ray_samples, ray_translation, ray_direction, z_vals, goal_pose, _ = data
        goal_pose = torch.stack([goal_pose[:, 38], goal_pose[:, 41]], axis=-1)

        # get values for coarse network and run them through the coarse network
        goal_pose_encoding_flat = self.human_pose_encoder.encode(goal_pose)
        goal_pose_encoding = goal_pose_encoding_flat[..., None, :].expand(goal_pose_encoding_flat.shape[0], ray_samples.shape[1],
                                                                     goal_pose_encoding_flat.shape[-1])

        samples_encoding = self.position_encoder.encode(ray_samples)
        warp_field_inputs = torch.cat([samples_encoding.reshape(-1, samples_encoding.shape[-1]),
                                       goal_pose_encoding.reshape(-1, goal_pose_encoding.shape[-1])], -1)
        warp = self.model_warp_field(warp_field_inputs).view(ray_samples.shape)

        warped_samples = ray_samples + warp
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
        rgb, weights = raw2outputs(raw_outputs, z_vals, coarse_samples_directions, self.args)
        if not self.args.run_fine:
            return rgb, rgb


        # get values for the fine network and run them through the fine network
        z_vals, ray_samples_fine = fine_sampling(ray_translation, ray_direction, z_vals, weights,
                                                 self.args.number_fine_samples)  # [batchsize, number_coarse_samples + number_fine_samples, 3]

        samples_encoding_fine = self.position_encoder.encode(ray_samples_fine)
        goal_pose_encoding = goal_pose_encoding_flat[..., None, :].expand(goal_pose_encoding_flat.shape[0],
                                                                          ray_samples_fine.shape[1],
                                                                          goal_pose_encoding_flat.shape[-1])
        warp_field_inputs_fine = torch.cat([samples_encoding_fine.reshape(-1, samples_encoding_fine.shape[-1]),
                                       goal_pose_encoding.reshape(-1, goal_pose_encoding.shape[-1])], -1)
        warp_fine = self.model_warp_field(warp_field_inputs_fine).view(ray_samples_fine.shape)

        warped_samples_fine = ray_samples_fine + warp_fine
        samples_encoding_fine = self.position_encoder.encode(warped_samples_fine)

        fine_samples_directions = warped_samples_fine - ray_translation[:, None,
                                                     :]  # [batchsize, number_coarse_samples, 3]
        samples_directions_norm_fine = fine_samples_directions / torch.norm(fine_samples_directions, dim=-1,
                                                                         keepdim=True)
        directions_encoding_fine = self.direction_encoder.encode(samples_directions_norm_fine)
        inputs_fine = torch.cat([samples_encoding_fine.view(-1, samples_encoding_fine.shape[-1]),
                                 directions_encoding_fine.reshape(-1, directions_encoding_fine.shape[-1])], -1)
        raw_outputs_fine = self.model_fine(
            inputs_fine)  # [batchsize * (number_coarse_samples + number_fine_samples), 4]
        raw_outputs_fine = raw_outputs_fine.reshape(samples_encoding_fine.shape[0], samples_encoding_fine.shape[1],
                                                    raw_outputs_fine.shape[
                                                        -1])  # [batchsize, number_coarse_samples + number_fine_samples, 4]
        # expand directions and translations to the number of coarse samples + fine_samples
        fine_samples_directions = ray_direction[..., None, :].expand(ray_direction.shape[0],
                                                                     ray_samples_fine.shape[1],
                                                                     ray_direction.shape[-1])
        rgb_fine, _ = raw2outputs(raw_outputs_fine, z_vals, fine_samples_directions, self.args)

        return rgb, rgb_fine
