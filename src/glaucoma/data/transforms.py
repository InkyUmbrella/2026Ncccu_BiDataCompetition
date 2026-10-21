import numpy as np
from PIL import Image
from torchvision import transforms as T


class Letterbox:
    """Fit the complete field of view in a square without changing aspect ratio."""

    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        scale = self.size / max(image.size)
        width, height = [max(1, min(self.size, round(v * scale))) for v in image.size]
        image = image.resize((width, height), Image.Resampling.BILINEAR)
        result = Image.new("RGB", (self.size, self.size))
        result.paste(image, ((self.size - width) // 2, (self.size - height) // 2))
        return result


class CropBlackBorder:
    """Keep the complete bright-field bounding box, with a conservative margin."""

    def __init__(self, threshold=10, margin=0.03):
        self.threshold, self.margin = threshold, margin

    def __call__(self, image):
        pixels = np.asarray(image)
        foreground = pixels.max(axis=2) > self.threshold
        rows = np.flatnonzero(foreground.any(axis=1))
        columns = np.flatnonzero(foreground.any(axis=0))
        if not len(rows) or not len(columns):
            return image
        left, right = int(columns[0]), int(columns[-1]) + 1
        top, bottom = int(rows[0]), int(rows[-1]) + 1
        # A tiny bright artifact on a nearly black image is not a reliable field.
        if (right - left) * (bottom - top) < image.width * image.height * 0.1:
            return image
        margin = round(max(right - left, bottom - top) * self.margin)
        return image.crop((max(0, left - margin), max(0, top - margin),
                           min(image.width, right + margin), min(image.height, bottom + margin)))


def preprocessing_spec(crop_black_border=False, crop_threshold=10, crop_margin=0.03):
    """Versioned cache identity; unused crop settings do not affect old caches."""
    if crop_black_border:
        return {"name": "crop_black_border_v1", "threshold": crop_threshold, "margin": crop_margin}
    return {"name": "letterbox_v1"}


def build_preprocess(size, crop_black_border=False, crop_threshold=10, crop_margin=0.03):
    operations = [CropBlackBorder(crop_threshold, crop_margin)] if crop_black_border else []
    return T.Compose(operations + [Letterbox(size)])


def build_transform(size, training=False, *, crop_black_border=False, crop_threshold=10,
                    crop_margin=0.03, preprocessed=False):
    # Cached images already passed through cropping and Letterbox exactly once.
    operations = [] if preprocessed else [build_preprocess(size, crop_black_border, crop_threshold, crop_margin)]
    if training:
        operations += [T.RandomHorizontalFlip(0.5),
                       T.RandomRotation(10, interpolation=T.InterpolationMode.BILINEAR, fill=0),
                       T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02)]
    return T.Compose(operations + [T.ToTensor(), T.Normalize((0.5,) * 3, (0.5,) * 3)])
