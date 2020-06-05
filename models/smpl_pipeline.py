import torch
from torch import nn
from utils import PositionalEncoder


class SmplPipeline(nn.Module):

    def __init__(self, model_coarse, args, position_encoder: PositionalEncoder,
                 direction_encoder: PositionalEncoder):
        super(SmplPipeline, self).__init__()
        self.args = args
        self.model_coarse = model_coarse
        self.position_encoder = position_encoder
        self.direction_encoder = direction_encoder

    def forward(self, data):
        """
            Pipeline for dataset with a single sample per ray.

            Returns
            -------
            rgb : torch.Tensor ([batch_size, 3])
                Estimated RGB color with coarse net.

            """
        ray_sample, ray_translation, samples_direction, warp, rgb_truth, goal_pose = data

        warped_sample = ray_sample + warp
        sample_encoding = self.position_encoder.encode(warped_sample)

        sample_direction = warped_sample - ray_translation  # [batchsize, 3]

        sample_direction_norm = sample_direction / torch.norm(sample_direction, dim=-1, keepdim=True)

        direction_encoding = self.direction_encoder.encode(sample_direction_norm)
        inputs = torch.cat([sample_encoding, direction_encoding], -1)  # [batchsize, encoding_size]
        raw_outputs = self.model_coarse(inputs)  # [batchsize, 4]
        rgb = torch.sigmoid(raw_outputs[..., :3])  # [batchsize, 3]
        return rgb, rgb
