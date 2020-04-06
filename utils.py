import numpy as np
import torch


def get_rays(H, W, focal, camera_transform):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i - W * .5) / focal, -(j - H * .5) / focal, -np.ones_like(i)], -1)
    rays_direction = np.sum(dirs[..., np.newaxis, :] * camera_transform[:3, :3], -1)  # dirs @ camera_transform
    rays_translation = np.broadcast_to(camera_transform[:3, -1], np.shape(rays_direction))
    return rays_translation, rays_direction


def positional_encoding(coordinate, number_frequencies, include_identity: bool):
    freq_bands = torch.pow(2, torch.linspace(0., number_frequencies - 1, number_frequencies))
    embed_fns = []
    if include_identity:
        embed_fns.append(lambda x: x)

    for freq in freq_bands:
        for periodic_fn in [torch.sin, torch.cos]:
            embed_fns.append(lambda x, periodic_fn=periodic_fn, freq=freq: periodic_fn(x * freq))

    return torch.cat([fn(coordinate) for fn in embed_fns], -1)

def raw2outputs(raw, z_vals, samples_directions, raw_noise_std=0.):
    raw2alpha = lambda raw, dists: 1. - torch.exp(-torch.nn.functional.relu(raw) * dists)

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(samples_directions[..., None, :], dim=-1)

    rgb = torch.nn.functional.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        !!!!!!!!!noise = torch.normal(0, raw_noise_std, raw[..., 3].shape)
    alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]
    weights = alpha * tf.math.cumprod(1. - alpha + 1e-10, -1, exclusive=True)
    rgb_map = tf.reduce_sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

    depth_map = tf.reduce_sum(weights * z_vals, -1)
    disp_map = 1. / tf.maximum(1e-10, depth_map / tf.reduce_sum(weights, -1))
    acc_map = tf.reduce_sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1. - acc_map[..., None])

    return rgb_map, disp_map, acc_map, weights, depth_map
