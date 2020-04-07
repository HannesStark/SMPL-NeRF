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


def raw2outputs(raw, z_vals, samples_directions, sigma_noise_std=0., white_background=False):
    raw2alpha = lambda raw, dists: 1. - torch.exp(-torch.nn.functional.relu(raw) * dists)

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], -1)  # [batchsize, number_samples]

    dists = dists * torch.norm(samples_directions[..., None, :], dim=-1)

    rgb = torch.nn.functional.sigmoid(raw[..., :3])  # [batchsize, number_samples, 3]
    noise = 0.
    if sigma_noise_std > 0.:
        noise = torch.normal(0, sigma_noise_std, raw[..., 3].shape)
    alpha = raw2alpha(raw[..., 3] + noise, dists)  # [batchsize, number_samples]
    one_minus_alpha = 1. - alpha + 1e-10

    # remove last column from one_minus_alhpa and add ones as first column so cumprod gives us the exclusive cumprod like tf.cumprod(exclusive=True)
    ones = torch.ones(one_minus_alpha.shape[:-1]).unsqueeze(-1)
    exclusive = torch.cat([ones, one_minus_alpha[..., :-1]], -1)
    weights = alpha * torch.cumprod(exclusive, -1)

    rgb = torch.sum(weights[..., None] * rgb, -2)  # [batchsize, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1. / torch.max(torch.full(depth_map.shape, 1e-10), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_background:
        rgb = rgb + (1. - acc_map[..., None])

    return rgb, weights


def fine_sampling(samples_translations, samples_directions, z_vals, weights, number_samples):
    z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
    z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], number_samples)
    z_samples = tf.stop_gradient(z_samples)

    z_vals = tf.sort(tf.concat([z_vals, z_samples], -1), -1)
    return samples_translations[..., None, :] + samples_directions[..., None, :] * z_vals[..., :,
                                                                                   None] # [batchsize, number_coarse_samples + number_fine_samples, 3]
