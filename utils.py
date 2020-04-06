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
