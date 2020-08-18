#!/usr/bin/env python3
import torch
import numpy as np
import cv2
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union, Callable
from torchvision.models import vgg16, vgg19
import torch.nn as nn
from torch.nn.modules.loss import _Loss

def img2mse(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Calculate MSE between two images.

    Parameters
    ----------
    x : torch.Tensor()
        Ground truth image.
    y : torch.Tensor()
        Generated image.

    Returns
    -------
    float
        MSE between x and y.

    """
    return torch.mean((x - y) ** 2)

def img2psnr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Calculate PSNR (Peak signal-to-noise ratio) Calculate MSE between
    two images (x, y).
    Parameters
    ----------
    x : torch.Tensor()
        Ground truth image with values between [0, 1].
    y : torch.Tensor()
        Generated image with values between [0, 1].

    Returns
    -------
    float
        PSNR between x and y..

    """
    mse = torch.mean((x - y) ** 2)
    return -10.*torch.log(mse)/torch.log(torch.Tensor([10.]))

def preprocess_image(im):
    """Resizes and shifts the dynamic range of image to 0-1
    Args:
        im: np.array, shape: (H, W, 3), dtype: float32 between 0-1 or np.uint8
    Return:
        im: torch.tensor, shape: (3, 299, 299), dtype: torch.float32 between 0-1
    """
    assert im.shape[2] == 3
    assert len(im.shape) == 3
    if im.dtype == np.uint8:
        im = im.astype(np.float32) / 255
    im = cv2.resize(im, (299, 299))
    im = np.rollaxis(im, axis=2)
    im = torch.from_numpy(im)
    assert im.max() <= 1.0
    assert im.min() >= 0.0
    assert im.dtype == torch.float32
    assert im.shape == (3, 299, 299)

    return im

def gaussian_filter(size: int, sigma: float) -> torch.Tensor:
    r"""Returns 2D Gaussian kernel N(0,`sigma`^2)
    Args:
        size: Size of the lernel
        sigma: Std of the distribution
    Returns:
        gaussian_kernel: 2D kernel with shape (1 x kernel_size x kernel_size)
    """
    coords = torch.arange(size).to(dtype=torch.float32)
    coords -= (size - 1) / 2.

    g = coords ** 2
    g = (- (g.unsqueeze(0) + g.unsqueeze(1)) / (2 * sigma ** 2)).exp()

    g /= g.sum()
    return g.unsqueeze(0)

def ssim(x: torch.Tensor, y: torch.Tensor, kernel_size: int = 11, kernel_sigma: float = 1.5,
         data_range: Union[int, float] = 1., reduction: str = 'mean', full: bool = False,
         k1: float = 0.01, k2: float = 0.03) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    r"""Interface of Structural Similarity (SSIM) index.
    Args:
        x: Batch of images. Required to be 2D (H, W), 3D (C,H,W), 4D (N,C,H,W) or 5D (N,C,H,W,2), channels first.
        y: Batch of images. Required to be 2D (H, W), 3D (C,H,W) 4D (N,C,H,W) or 5D (N,C,H,W,2), channels first.
        kernel_size: The side-length of the sliding window used in comparison. Must be an odd value.
        kernel_sigma: Sigma of normal distribution.
        data_range: Value range of input images (usually 1.0 or 255).
        reduction: Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
        full: Return cs map or not.
        k1: Algorithm parameter, K1 (small constant, see [1]).
        k2: Algorithm parameter, K2 (small constant, see [1]).
            Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
    Returns:
        Value of Structural Similarity (SSIM) index. In case of 5D input tensors, complex value is returned
        as a tensor of size 2.
    References:
        .. [1] Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P.
           (2004). Image quality assessment: From error visibility to
           structural similarity. IEEE Transactions on Image Processing,
           13, 600-612.
           https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf,
           :DOI:`10.1109/TIP.2003.819861`
    """
    

    kernel = gaussian_filter(kernel_size, kernel_sigma).repeat(x.size(1), 1, 1, 1).to(y)
    ssim_map, cs_map = _ssim_per_channel(x=x, y=y, kernel=kernel, data_range=data_range, k1=k1, k2=k2)
    ssim_val = ssim_map.mean(1)
    cs = cs_map.mean(1)

    if reduction != 'none':
        reduction_operation = {'mean': torch.mean,
                               'sum': torch.sum}
        ssim_val = reduction_operation[reduction](ssim_val, dim=0)
        cs = reduction_operation[reduction](cs, dim=0)

    if full:
        return ssim_val, cs

    return ssim_val
def _ssim_per_channel(x: torch.Tensor, y: torch.Tensor, kernel: torch.Tensor,
                      data_range: Union[float, int] = 1., k1: float = 0.01,
                      k2: float = 0.03) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    r"""Calculate Structural Similarity (SSIM) index for X and Y per channel.
    Args:
        x: Batch of images, (N,C,H,W).
        y: Batch of images, (N,C,H,W).
        kernel: 2D Gaussian kernel.
        data_range: Value range of input images (usually 1.0 or 255).
        k1: Algorithm parameter, K1 (small constant, see [1]).
        k2: Algorithm parameter, K2 (small constant, see [1]).
            Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
    Returns:
        Full Value of Structural Similarity (SSIM) index.
    """

    if x.size(-1) < kernel.size(-1) or x.size(-2) < kernel.size(-2):
        raise ValueError(f'Kernel size can\'t be greater than actual input size. Input size: {x.size()}. '
                         f'Kernel size: {kernel.size()}')

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    n_channels = x.size(1)
    mu1 = F.conv2d(x, weight=kernel, stride=1, padding=0, groups=n_channels)
    mu2 = F.conv2d(y, weight=kernel, stride=1, padding=0, groups=n_channels)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    compensation = 1.0
    sigma1_sq = compensation * (F.conv2d(x * x, weight=kernel, stride=1, padding=0, groups=n_channels) - mu1_sq)
    sigma2_sq = compensation * (F.conv2d(y * y, weight=kernel, stride=1, padding=0, groups=n_channels) - mu2_sq)
    sigma12 = compensation * (F.conv2d(x * y, weight=kernel, stride=1, padding=0, groups=n_channels) - mu1_mu2)

    # Set alpha = beta = gamma = 1.
    cs_map = (2 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2)
    ssim_map = ((2 * mu1_mu2 + c1) / (mu1_sq + mu2_sq + c1)) * cs_map

    ssim_val = ssim_map.mean(dim=(-1, -2))
    cs = cs_map.mean(dim=(-1, -2))
    return ssim_val, cs

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Constant used in feature normalization to avoid zero division
EPS = 1e-10

# Map VGG names to corresponding number in torchvision layer
VGG16_LAYERS = {
    "conv1_1": '0', "relu1_1": '1',
    "conv1_2": '2', "relu1_2": '3',
    "pool1": '4',
    "conv2_1": '5', "relu2_1": '6',
    "conv2_2": '7', "relu2_2": '8',
    "pool2": '9',
    "conv3_1": '10', "relu3_1": '11',
    "conv3_2": '12', "relu3_2": '13',
    "conv3_3": '14', "relu3_3": '15',
    "pool3": '16',
    "conv4_1": '17', "relu4_1": '18',
    "conv4_2": '19', "relu4_2": '20',
    "conv4_3": '21', "relu4_3": '22',
    "pool4": '23',
    "conv5_1": '24', "relu5_1": '25',
    "conv5_2": '26', "relu5_2": '27',
    "conv5_3": '28', "relu5_3": '29',
    "pool5": '30',
}

VGG19_LAYERS = {
    "conv1_1": '0', "relu1_1": '1',
    "conv1_2": '2', "relu1_2": '3',
    "pool1": '4',
    "conv2_1": '5', "relu2_1": '6',
    "conv2_2": '7', "relu2_2": '8',
    "pool2": '9',
    "conv3_1": '10', "relu3_1": '11',
    "conv3_2": '12', "relu3_2": '13',
    "conv3_3": '14', "relu3_3": '15',
    "conv3_4": '16', "relu3_4": '17',
    "pool3": '18',
    "conv4_1": '19', "relu4_1": '20',
    "conv4_2": '21', "relu4_2": '22',
    "conv4_3": '23', "relu4_3": '24',
    "conv4_4": '25', "relu4_4": '26',
    "pool4": '27',
    "conv5_1": '28', "relu5_1": '29',
    "conv5_2": '30', "relu5_2": '31',
    "conv5_3": '32', "relu5_3": '33',
    "conv5_4": '34', "relu5_4": '35',
    "pool5": '36',
}

def _adjust_dimensions(input_tensors: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]):
    r"""Expands input tensors dimensions to 4D
    """
    if isinstance(input_tensors, torch.Tensor):
        input_tensors = (input_tensors,)

    resized_tensors = []
    for tensor in input_tensors:
        tmp = tensor.clone()
        if tmp.dim() == 2:
            tmp = tmp.unsqueeze(0)
        if tmp.dim() == 3:
            tmp = tmp.unsqueeze(0)
        if tmp.dim() != 4 and tmp.dim() != 5:
            raise ValueError(f'Expected 2, 3, 4 or 5 dimensions (got {tensor.dim()})')
        resized_tensors.append(tmp)

    if len(resized_tensors) == 1:
        return resized_tensors[0]

    return tuple(resized_tensors)

def _validate_input(
        input_tensors: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        allow_5d: bool,
        kernel_size: Optional[int] = None,
        scale_weights: Union[Optional[Tuple[float]], Optional[List[float]], Optional[torch.Tensor]] = None) -> None:

    if isinstance(input_tensors, torch.Tensor):
        input_tensors = (input_tensors,)

    assert isinstance(input_tensors, tuple)
    assert 0 < len(input_tensors) < 3, f'Expected one or two input tensors, got {len(input_tensors)}'

    min_n_dim = 2
    max_n_dim = 5 if allow_5d else 4
    for tensor in input_tensors:
        assert isinstance(tensor, torch.Tensor), f'Expected input to be torch.Tensor, got {type(tensor)}.'
        assert min_n_dim <= tensor.dim() <= max_n_dim, \
            f'Input images must be {min_n_dim}D - {max_n_dim}D tensors, got images of shape {tensor.size()}.'
        assert torch.all(tensor >= 0), 'Expected input tensor greater or equal 0'
        if tensor.dim() == 5:
            assert tensor.size(-1) == 2, f'Expected Complex 5D tensor with (N,C,H,W,2) size, got {tensor.size()}'

    if len(input_tensors) == 2:
        assert input_tensors[0].size() == input_tensors[1].size(), \
            f'Input images must have the same dimensions, got {input_tensors[0].size()} and {input_tensors[1].size()}.'

    if kernel_size is not None:
        assert kernel_size % 2 == 1, f'Kernel size must be odd, got {kernel_size}.'
    if scale_weights is not None:
        assert isinstance(scale_weights, (list, tuple, torch.Tensor)), \
            f'Scale weights must be of type list, tuple or torch.Tensor, got {type(scale_weights)}.'
        if isinstance(scale_weights, (list, tuple)):
            scale_weights = torch.tensor(scale_weights)
        assert (scale_weights.dim() == 1), \
            f'Scale weights must be one dimensional, got {scale_weights.dim()}.'


class ContentLoss(_Loss):
    r"""Creates Content loss that can be used for image style transfer of as a measure in
    image to image tasks.
    Uses pretrained VGG models from torchvision. Normalizes features before summation.
    Expects input to be in range [0, 1] or normalized with ImageNet statistics into range [-1, 1]
    Args:
        feature_extractor: Model to extract features or model name in {`vgg16`, `vgg19`}.
        layers: List of strings with layer names. Default: [`relu3_3`]
        weights: List of float weight to balance different layers
        replace_pooling: Flag to replace MaxPooling layer with AveragePooling. See [1] for details.
        distance: Method to compute distance between features. One of {`mse`, `mae`}.
        reduction: Reduction over samples in batch: "mean"|"sum"|"none"
        mean: List of float values used for data standartization. Default: ImageNet mean.
            If there is no need to normalize data, use [0., 0., 0.].
        std: List of float values used for data standartization. Default: ImageNet std.
            If there is no need to normalize data, use [1., 1., 1.].
        normalize_features: If true, unit-normalize each feature in channel dimension before scaling
            and computing distance. See [2] for details.
    References:
        .. [1] Gatys, Leon and Ecker, Alexander and Bethge, Matthias
        (2016). A Neural Algorithm of Artistic Style}
        Association for Research in Vision and Ophthalmology (ARVO)
        https://arxiv.org/abs/1508.06576
        .. [2] Zhang, Richard and Isola, Phillip and Efros, et al.
        (2018) The Unreasonable Effectiveness of Deep Features as a Perceptual Metric
        2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition
        https://arxiv.org/abs/1801.03924
    """

    def __init__(self, feature_extractor: Union[str, Callable] = "vgg16", layers: Tuple[str] = ("relu3_3", ),
                 weights: List[Union[float, torch.Tensor]] = [1.], replace_pooling: bool = False,
                 distance: str = "mse", reduction: str = "mean", mean: List[float] = IMAGENET_MEAN,
                 std: List[float] = IMAGENET_STD, normalize_features: bool = False) -> None:

        super().__init__()

        if callable(feature_extractor):
            self.model = feature_extractor
            self.layers = layers
        else:
            if feature_extractor == "vgg16":
                self.model = vgg16(pretrained=True, progress=False).features
                self.layers = [VGG16_LAYERS[l] for l in layers]
            elif feature_extractor == "vgg19":
                self.model = vgg19(pretrained=True, progress=False).features
                self.layers = [VGG19_LAYERS[l] for l in layers]
            else:
                raise ValueError("Unknown feature extractor")

        if replace_pooling:
            self.model = self.replace_pooling(self.model)

        # Disable gradients
        for param in self.model.parameters():
            param.requires_grad_(False)

        self.distance = {
            "mse": nn.MSELoss,
            "mae": nn.L1Loss,
        }[distance](reduction='none')

        self.weights = [torch.tensor(w) for w in weights]
        mean = torch.tensor(mean)
        std = torch.tensor(std)
        self.mean = mean.view(1, -1, 1, 1)
        self.std = std.view(1, -1, 1, 1)
        
        self.normalize_features = normalize_features
        self.reduction = reduction

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        r"""Computation of Content loss between feature representations of prediction and target tensors.
        Args:
            prediction: Tensor of prediction of the network.
            target: Reference tensor.
        """
        _validate_input(input_tensors=(prediction, target), allow_5d=False)
        prediction, target = _adjust_dimensions(input_tensors=(prediction, target))

        self.model.to(prediction)
        prediction_features = self.get_features(prediction)
        target_features = self.get_features(target)

        distances = self.compute_distance(prediction_features, target_features)

        # Scale distances, then average in spatial dimensions, then stack and sum in channels dimension
        loss = torch.cat([(d * w.to(d)).mean(dim=[2, 3]) for d, w in zip(distances, self.weights)], dim=1).sum(dim=1)

        if self.reduction == 'none':
            return loss

        return {'mean': loss.mean,
                'sum': loss.sum
                }[self.reduction](dim=0)

    def compute_distance(self, prediction_features: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
        r"""Take L2 or L1 distance between feature maps"""
        return [self.distance(x, y) for x, y in zip(prediction_features, target_features)]

    def get_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        r"""
        Args:
            x: torch.Tensor with shape (N, C, H, W)
        
        Returns:
            features: List of features extracted from intermediate layers
        """
        # Normalize input
        x = (x - self.mean.to(x)) / self.std.to(x)

        features = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.layers:
                features.append(self.normalize(x) if self.normalize_features else x)
        return features

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        r"""Normalize feature maps in channel direction to unit length.
        Args:
            x: Tensor with shape (N, C, H, W)
        Returns:
            x_norm: Normalized input
        """
        norm_factor = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))
        return x / (norm_factor + EPS)

    def replace_pooling(self, module: torch.nn.Module) -> torch.nn.Module:
        r"""Turn All MaxPool layers into AveragePool"""
        module_output = module
        if isinstance(module, torch.nn.MaxPool2d):
            module_output = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
            
        for name, child in module.named_children():
            module_output.add_module(name, self.replace_pooling(child))
        return module_output

class LPIPS(ContentLoss):
    r"""Learned Perceptual Image Patch Similarity metric.
    For now only VGG16 learned weights are supported.
    Expects input to be in range [0, 1] or normalized with ImageNet statistics into range [-1, 1]
    Args:
        feature_extractor: Name of model used to extract features. One of {`vgg16`, `vgg19`}
        use_average_pooling: Flag to replace MaxPooling layer with AveragePooling. See [1] for details.
        distance: Method to compute distance between features. One of {`mse`, `mae`}.
        reduction: Reduction over samples in batch: "mean"|"sum"|"none"
        mean: List of float values used for data standartization. Default: ImageNet mean.
            If there is no need to normalize data, use [0., 0., 0.].
        std: List of float values used for data standartization. Default: ImageNet std.
            If there is no need to normalize data, use [1., 1., 1.].
    References:
        .. [1] Gatys, Leon and Ecker, Alexander and Bethge, Matthias
        (2016). A Neural Algorithm of Artistic Style}
        Association for Research in Vision and Ophthalmology (ARVO)
        https://arxiv.org/abs/1508.06576
        .. [2] Zhang, Richard and Isola, Phillip and Efros, et al.
        (2018) The Unreasonable Effectiveness of Deep Features as a Perceptual Metric
        2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition
        https://arxiv.org/abs/1801.03924
    """
    _weights_url = "https://github.com/photosynthesis-team/" + \
        "photosynthesis.metrics/releases/download/v0.4.0/lpips_weights.pt"

    def __init__(self, replace_pooling: bool = False, distance: str = "mse", reduction: str = "mean",
                 mean: List[float] = IMAGENET_MEAN, std: List[float] = IMAGENET_STD,) -> None:
        lpips_layers = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3']
        lpips_weights = torch.hub.load_state_dict_from_url(self._weights_url, progress=False)
        super().__init__("vgg16", layers=lpips_layers, weights=lpips_weights,
                         replace_pooling=replace_pooling, distance=distance,
                         reduction=reduction, mean=mean, std=std,
                         normalize_features=True)
def print_scores(x, y):
    mse_score = img2mse(x, y)
    psnr_score = img2psnr(x, y)
    ssim_score = ssim(x, y)
    lpips_loss = LPIPS()
    lpips_score = lpips_loss(x, y)
    print("MSE: {}, PSNR: {:.3f}, SSIM: {:.3f}, LPIPS: {:.3f}".format(
        mse_score.item(), psnr_score.item(), ssim_score.item(), lpips_score.item()))


if __name__ == "__main__":
    x = torch.Tensor([cv2.imread("img_000.png")/255.])
    y = torch.Tensor([cv2.imread("img_001.png")/255.])
    x = x.permute(0, 3, 1, 2)
    y = y.permute(0, 3, 1, 2)
    mse_score = img2mse(x, y)
    psnr_score = img2psnr(x, y)
    ssim_score = ssim(x, y)
    lpips_loss = LPIPS()
    lpips_score = lpips_loss(x, y)
    print("X shape: {}, Y shape: {}".format(x.shape, y.shape))
    print("MSE: {}, PSNR: {:.3f}, SSIM: {:.3f}, LPIPS: {:.3f}".format(
        mse_score.item(), psnr_score.item(), ssim_score.item(), lpips_score.item()))
