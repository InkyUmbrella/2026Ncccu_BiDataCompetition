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


def load_rgb(path):
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.load()
        return image


class FundusDataset(Dataset):
    def __init__(self, samples, age_stats, transform):
        self.samples = samples
        self.age_stats = age_stats
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        try:
            image = self.transform(load_rgb(sample.path))
        except Exception as exc:
            raise RuntimeError(f"Cannot decode/transform {sample.path}: {exc}") from exc
        result = {"image": image,
                  "age": torch.tensor(self.age_stats.transform(sample.age), dtype=torch.float32),
                  "raw_age": sample.age, "name": sample.name}
        if sample.label is not None:
            result["label"] = torch.tensor(sample.label, dtype=torch.float32)
        return result
