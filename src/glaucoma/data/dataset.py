import csv
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset

from .age import parse_age


@dataclass(frozen=True)
class Sample:
    name: str
    path: Path
    age: float
    label: int | None


def read_samples(root, split, require_labels=True):
    if split not in ("train", "valid", "test"):
        raise ValueError(f"Unknown split: {split}")
    root = Path(root)
    files = {}
    for path in sorted((root / split).iterdir()):
        if path.is_file():
            key = path.name.casefold()
            if key in files:
                raise ValueError(f"Ambiguous case-insensitive filename: {files[key]} / {path}")
            files[key] = path
    samples, seen = [], set()
    with (root / f"{split}.csv").open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"Image_name", "Age"} | ({"True_Label"} if require_labels else set())
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{split}.csv requires columns {sorted(required)}")
        for line, row in enumerate(reader, start=2):
            name = row["Image_name"]
            if not name or "/" in name or "\\" in name or name in (".", ".."):
                raise ValueError(f"Invalid Image_name at {split}.csv:{line}")
            key = name.casefold()
            if key in seen:
                raise ValueError(f"Duplicate CSV image name: {name}")
            seen.add(key)
            if key not in files:
                raise FileNotFoundError(f"{split}.csv:{line}: image not found: {name}")
            label = None
            if require_labels:
                if row["True_Label"] not in ("0", "1"):
                    raise ValueError(f"Invalid binary label at {split}.csv:{line}")
                label = int(row["True_Label"])
            samples.append(Sample(name, files[key], parse_age(row["Age"]), label))
    if not samples:
        raise ValueError(f"Empty dataset: {split}")
    return samples


DECODE_POLICIES = ("full", "draft")


def load_rgb(path, decode="full", image_size=384):
    """Decode an image to RGB, optionally JPEG-drafting to the model input size.

    ``draft`` must be requested before ``exif_transpose``: loading or transposing
    first makes Pillow's draft call a silent no-op. The decoded short side stays
    >= ``image_size``, so the letterbox never has to upscale, and non-JPEG
    formats are decoded exactly as before.
    """
    if decode not in DECODE_POLICIES:
        raise ValueError(f"Unknown decode policy: {decode}")
    if image_size < 1:
        raise ValueError("image_size must be positive")
    with Image.open(path) as image:
        if decode != "full" and image.format == "JPEG":
            try:
                image.draft("RGB", (image_size, image_size))
            except (ValueError, OSError, AttributeError):
                pass  # Unusual JPEG modes fall back to the full decode path.
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.load()
        if decode != "full" and max(image.size) > 4 * image_size:
            # Non-JPEG inputs ignore draft; bound memory for pixel-bomb files.
            image.thumbnail((4 * image_size, 4 * image_size), Image.Resampling.BILINEAR)
        return image


class FundusDataset(Dataset):
    def __init__(self, samples, age_stats, transform, image_cache=None, decode="full", image_size=384):
        self.samples = samples
        self.age_stats = age_stats
        self.transform = transform
        self.image_cache = image_cache
        self.decode = decode
        self.image_size = image_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        try:
            image = self.transform(
                self.image_cache.load(sample.path) if self.image_cache
                else load_rgb(sample.path, self.decode, self.image_size))
        except Exception as exc:
            raise RuntimeError(f"Cannot decode/transform {sample.path}: {exc}") from exc
        result = {"image": image,
                  "age": torch.tensor(self.age_stats.transform(sample.age), dtype=torch.float32),
                  "raw_age": sample.age, "name": sample.name}
        if sample.label is not None:
            result["label"] = torch.tensor(sample.label, dtype=torch.float32)
        return result
