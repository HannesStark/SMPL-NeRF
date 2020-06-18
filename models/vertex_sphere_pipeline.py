import torch

from models.nerf_pipeline import NerfPipeline
from utils import PositionalEncoder, raw2outputs, fine_sampling


class VertexSpherePipeline(NerfPipeline):

    def __init__(self, model_coarse, model_fine, args, position_encoder: PositionalEncoder,
                 direction_encoder: PositionalEncoder):
        super(VertexSpherePipeline, self).__init__(model_coarse, model_fine, args, position_encoder, direction_encoder)
        self.args = args

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
        ray_samples, ray_translation, ray_direction, z_vals, warp, _ = data
        # get values for coarse network and run them through the coarse network

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
        rgb, weights, densities = raw2outputs(raw_outputs, z_vals, coarse_samples_directions, self.args)
        # Take mean over samples for rgb value debugging
        #rgb = torch.sigmoid(raw_outputs[..., :3])
        #rgb = torch.mean(rgb, 1)

        if not self.args.run_fine:
            return rgb, rgb, warp, ray_samples, warped_samples, densities

        raise NotImplementedError('calculating the deterministic/true warp for the fine samples in not implemented yet')
#
## get values for the fine network and run them through the fine network
# z_vals, ray_samples_fine = fine_sampling(ray_translation, ray_direction, z_vals, weights,
#                                         self.args.number_fine_samples)  # [batchsize, number_coarse_samples + number_fine_samples, 3]
#
# warp_fine =  TODO
#
# warped_samples_fine = ray_samples_fine + warp_fine
# samples_encoding_fine = self.position_encoder.encode(warped_samples_fine)
#
# fine_samples_directions = warped_samples_fine - ray_translation[:, None,
#                                                :]  # [batchsize, number_coarse_samples, 3]
# samples_directions_norm_fine = fine_samples_directions / torch.norm(fine_samples_directions, dim=-1,
#                                                                    keepdim=True)
# directions_encoding_fine = self.direction_encoder.encode(samples_directions_norm_fine)
# inputs_fine = torch.cat([samples_encoding_fine.view(-1, samples_encoding_fine.shape[-1]),
#                         directions_encoding_fine.reshape(-1, directions_encoding_fine.shape[-1])], -1)
# raw_outputs_fine = self.model_fine(
#    inputs_fine)  # [batchsize * (number_coarse_samples + number_fine_samples), 4]
# raw_outputs_fine = raw_outputs_fine.reshape(samples_encoding_fine.shape[0], samples_encoding_fine.shape[1],
#                                            raw_outputs_fine.shape[
#                                                -1])  # [batchsize, number_coarse_samples + number_fine_samples, 4]
## expand directions and translations to the number of coarse samples + fine_samples
# fine_samples_directions = ray_direction[..., None, :].expand(ray_direction.shape[0],
#                                                             ray_samples_fine.shape[1],
#                                                             ray_direction.shape[-1])
# rgb_fine, weights, densities_fine = raw2outputs(raw_outputs_fine, z_vals, fine_samples_directions, self.args)
#
# return rgb, rgb_fine, warp_fine, ray_samples_fine, warped_samples_fine, densities_fine
