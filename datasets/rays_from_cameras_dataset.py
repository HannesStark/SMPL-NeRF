import numpy as np
from torch.utils.data import Dataset

from utils import get_rays


class RaysFromCamerasDataset(Dataset):
    """
    Dataset of rays for without ground truth images (used for inference).
    """

    def __init__(self, camera_transforms: np.array, height: int, widht: int,
                 focal: float, transform) -> None:
        """
        Parameters
        ----------
        camera_transforms : [np.array]
            List of camera transformations for inference.
        height : int
            Height of image.
        widht : int
            Width of image.
        focal : float
            Focal length of camera.
        transform :
            List of callable transforms for preprocessing.
        """
        super().__init__()
        self.transform = transform
        self.rays = []  # list of arrays with ray translation, ray direction and rgb
        print('Start initializing all rays of all images')
        for transform in camera_transforms:
            rays_translation, rays_direction = get_rays(height, widht, focal, transform)
            trans_dir_stack = np.stack([rays_translation, rays_direction], -2)
            trans_dir_list = trans_dir_stack.reshape((-1, 2, 3))
            self.rays.append(trans_dir_list)
        self.rays = np.concatenate(self.rays)
        print('Finish initializing rays')

    def __getitem__(self, index: int):
        """
        Takes ray of given index and returns transformed ray
        """
        rays_translation, rays_direction = self.rays[index]
        ray_samples, samples_translations, samples_directions, z_vals, _ = self.transform(
            (rays_translation, rays_direction, []))

        return ray_samples, samples_translations, samples_directions, z_vals

    def __len__(self) -> int:
        return len(self.rays)
