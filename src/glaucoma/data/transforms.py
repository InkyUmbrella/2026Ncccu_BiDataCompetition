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


def build_transform(size, training=False):
    operations = [Letterbox(size)]
    if training:
        operations += [T.RandomHorizontalFlip(0.5),
                       T.RandomRotation(10, interpolation=T.InterpolationMode.BILINEAR, fill=0),
                       T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02)]
    return T.Compose(operations + [T.ToTensor(), T.Normalize((0.5,) * 3, (0.5,) * 3)])
