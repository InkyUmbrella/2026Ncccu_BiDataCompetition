"""Two-view augmentation for MoCo-v2 on already cropped/letterboxed fundus images.

The cache is produced by ``build_preprocess`` (crop + letterbox), so this module
must never crop black borders or letterbox again. It also re-uses the fixed
``mean=std=0.5`` normalisation of the supervised baseline, so SSL introduces no
data-dependent statistic.
"""

from PIL import ImageOps
import torch
from torchvision import transforms as T

AUGMENTATION_SPEC = "fundus_moco_v2_v1"


class RandomSolarize:
    """Solarise an RGB PIL image with probability ``p``.

    Implemented locally (PIL is already a dependency) so the recipe does not
    depend on the torchvision version. The draw uses the global torch RNG, so
    seeding with ``torch.manual_seed`` makes the whole two-view pipeline
    reproducible.
    """

    def __init__(self, threshold=128, p=0.2):
        if not 0 <= threshold <= 255:
            raise ValueError("threshold must be in [0, 255]")
        if not 0 <= p <= 1:
            raise ValueError("p must be in [0, 1]")
        self.threshold = threshold
        self.p = p

    def __call__(self, image):
        if self.p and torch.rand(1).item() < self.p:
            return ImageOps.solarize(image, self.threshold)
        return image


def build_ssl_transform(size, *, min_crop_scale=0.2):
    """Return a callable producing two independently augmented views.

    ``FundusDataset`` stores the returned tuple under ``"image"``; the default
    collate turns ``[(v1, v2), ...]`` into ``[batch_v1, batch_v2]``.
    """
    if not isinstance(size, int) or size < 32:
        raise ValueError("size must be an integer >= 32")
    if not 0 < min_crop_scale <= 1:
        raise ValueError("min_crop_scale must be in (0, 1]")
    view = T.Compose([
        T.RandomResizedCrop(size, scale=(min_crop_scale, 1.0), ratio=(0.75, 4 / 3),
                            antialias=True),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(15, interpolation=T.InterpolationMode.BILINEAR, fill=0),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        T.RandomGrayscale(p=0.2),
        T.RandomApply([T.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))], p=0.3),
        RandomSolarize(threshold=128, p=0.2),
        T.ToTensor(),
        T.Normalize((0.5,) * 3, (0.5,) * 3),
    ])
    return TwoViewTransform(view)


class TwoViewTransform:
    """Apply the same random pipeline twice, independently, to one image."""

    def __init__(self, view_transform):
        self.view_transform = view_transform

    def __call__(self, image):
        return self.view_transform(image), self.view_transform(image)


def build_probe_transform(size, *, crop_black_border=False, crop_threshold=10,
                          crop_margin=0.03, preprocessed=False):
    """Deterministic transform for frozen linear probes.

    Cache entries already passed through crop/letterbox exactly once; the plan's
    gate extracts features at the downstream 384 resolution.
    """
    operations = []
    if not preprocessed:
        from ..data.transforms import build_preprocess
        operations.append(build_preprocess(size, crop_black_border, crop_threshold, crop_margin))
    if size != 384:
        operations.append(T.Resize(size, antialias=True))
    return T.Compose(operations + [T.ToTensor(), T.Normalize((0.5,) * 3, (0.5,) * 3)])
