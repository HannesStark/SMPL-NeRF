import numpy as np
import torch

from torchsearchsorted import searchsorted


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

    dists = dists * torch.norm(samples_directions, dim=-1)

    rgb = torch.sigmoid(raw[..., :3])  # [batchsize, number_samples, 3]
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


def sample_pdf(bins, weights, number_samples):
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    u = torch.linspace(0., 1., steps=number_samples)
    u = u.expand(list(cdf.shape[:-1]) + [number_samples])

    # Invert CDF
    u = u.contiguous()
    inds = searchsorted(cdf, u, side='right')
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min(cdf.shape[-1] - 1 * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


def fine_sampling(samples_translations, samples_directions, z_vals, weights, number_samples):
    z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
    z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], number_samples)
    z_samples = z_samples.detach()

    z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
    return samples_translations[..., None, :] + samples_directions[..., None, :] * z_vals[..., :,
                                                                                   None]  # [batchsize, number_coarse_samples + number_fine_samples, 3]


def run_nerf_pipeline(ray_samples, samples_translations, samples_directions, z_vals, model_coarse, model_fine,
                      sigma_noise_std, number_fine_samples, white_background):
    # get values for coarse network and run them through the coarse network
    samples_encoding = positional_encoding(ray_samples, 10, True)
    samples_directions_norm = samples_directions / torch.norm(samples_directions, dim=-1, keepdim=True)
    directions_encoding = positional_encoding(samples_directions_norm, 4, False)
    # flatten the encodings from [batchsize, number_coarse_samples, encoding_size] to [batchsize * number_coarse_samples, encoding_size] and concatenate
    inputs = torch.cat([samples_encoding.view(-1, samples_encoding.shape[-1]),
                        directions_encoding.view(-1, directions_encoding.shape[-1])], -1)
    raw_outputs = model_coarse(inputs)  # [batchsize * number_coarse_samples, 4]
    raw_outputs = raw_outputs.view(samples_encoding.shape[0], samples_encoding.shape[1],
                                   raw_outputs.shape[-1])  # [batchsize, number_coarse_samples, 4]
    rgb, weights = raw2outputs(raw_outputs, z_vals, samples_directions, sigma_noise_std, white_background)

    # get values for the fine network and run them through the fine network
    ray_samples_fine = fine_sampling(samples_translations, samples_directions, z_vals, weights,
                                     number_fine_samples)  # [batchsize, number_coarse_samples + number_fine_samples, 3]
    samples_encoding_fine = positional_encoding(ray_samples_fine, 10, True)
    # expand directions and translations to the number of coarse samples + fine_samples
    directions_encoding_fine = directions_encoding[..., :1, :].expand(directions_encoding.shape[0],
                                                                      ray_samples_fine.shape[1],
                                                                      directions_encoding.shape[-1])
    inputs_fine = torch.cat([samples_encoding_fine.view(-1, samples_encoding_fine.shape[-1]),
                             directions_encoding_fine.view(-1, samples_encoding_fine.shape[-1])], -1)
    raw_outputs_fine = model_fine(inputs_fine)  # [batchsize * (number_coarse_samples + number_fine_samples), 4]
    raw_outputs_fine = raw_outputs.view(samples_encoding_fine.shape[0], samples_encoding_fine.shape[1],
                                        raw_outputs_fine.shape[
                                            -1])  # [batchsize, number_coarse_samples + number_fine_samples, 4]
    # expand directions and translations to the number of coarse samples + fine_samples
    samples_directions_fine = samples_directions[..., :1, :].expand(samples_directions.shape[0],
                                                                    ray_samples_fine.shape[1],
                                                                    samples_directions.shape[-1])
    rgb_fine, _ = raw2outputs(raw_outputs_fine, z_vals, samples_directions_fine, sigma_noise_std, white_background)

    return rgb, rgb_fine
