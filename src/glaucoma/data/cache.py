"""Lossless cache of deterministic RGB/letterbox preprocessing.

Random augmentation stays in the dataset and is applied afresh each epoch.
"""

from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from .dataset import load_rgb, read_samples
from .transforms import build_preprocess, preprocessing_spec
from ..utils import atomic_write, save_json, sha256_file


def _cache_one(task):
    source, target, size, crop, threshold, margin = task
    source = Path(source)
    before = source.stat()
    pixels = np.asarray(build_preprocess(size, crop, threshold, margin)(load_rgb(source)))
    atomic_write(target, lambda handle: np.save(handle, pixels, allow_pickle=False))
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"Source changed while caching: {source}")
    return source.parent.name + "/" + source.name, [after.st_size, after.st_mtime_ns]


def prepare_cache(data_root, output, image_size=384, workers=4, *, crop_black_border=False,
                  crop_threshold=10, crop_margin=0.03):
    if image_size < 32 or workers < 1:
        raise ValueError("image_size must be >=32 and workers must be positive")
    from ..config import Config
    Config.from_dict({"data": {"image_size": image_size, "crop_black_border": crop_black_border,
                              "crop_threshold": crop_threshold, "crop_margin": crop_margin}})
    root, output = Path(data_root).resolve(), Path(output).resolve()
    rows = [row for split in ("train", "valid") for row in read_samples(root, split)]
    output.mkdir(parents=True, exist_ok=False)
    estimate = len(rows) * (image_size * image_size * 3 + 256)
    if shutil.disk_usage(output).free < estimate * 1.1:
        raise ValueError(f"Cache requires approximately {estimate / 1024**3:.1f} GiB plus headroom")
    hashes = {split: sha256_file(root / f"{split}.csv") for split in ("train", "valid")}
    tasks = [(str(row.path), str(output / row.path.parent.name / (row.path.name + ".npy")),
              image_size, crop_black_border, crop_threshold, crop_margin)
             for row in rows]
    entries = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for index, (key, signature) in enumerate(pool.map(_cache_one, tasks, chunksize=8), 1):
            entries[key] = signature
            if index % 1000 == 0 or index == len(rows):
                print(f"[cache] {index}/{len(rows)}", flush=True)
    for split, digest in hashes.items():
        if sha256_file(root / f"{split}.csv") != digest:
            raise RuntimeError(f"CSV changed while caching: {split}")
    save_json(output / "manifest.json", {"version": 2, "source_root": str(root),
              "preprocessing": preprocessing_spec(crop_black_border, crop_threshold, crop_margin),
              "image_size": image_size, "csv_hashes": hashes, "entries": entries})
    print(f"[cache] complete: {output}", flush=True)


class ImageCache:
    def __init__(self, directory, samples, image_size, *, crop_black_border=False,
                 crop_threshold=10, crop_margin=0.03):
        self.directory = Path(directory)
        manifest = json.loads((self.directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest["version"] not in (1, 2) or manifest["image_size"] != image_size:
            raise ValueError("Image cache version or image size mismatch")
        stored = {"name": "letterbox_v1"} if manifest["version"] == 1 else manifest["preprocessing"]
        if stored != preprocessing_spec(crop_black_border, crop_threshold, crop_margin):
            raise ValueError("Image cache preprocessing mismatch; rebuild from original images")
        self.image_size = image_size
        roots = {row.path.parent.parent.resolve() for row in samples}
        if roots != {Path(manifest["source_root"])}:
            raise ValueError("Image cache source root mismatch")
        for split in {row.path.parent.name for row in samples}:
            root = next(iter(roots))
            if sha256_file(root / f"{split}.csv") != manifest["csv_hashes"][split]:
                raise ValueError(f"Image cache CSV mismatch: {split}")
        for row in samples:
            stat = row.path.stat()
            key = row.path.parent.name + "/" + row.path.name
            if manifest["entries"].get(key) != [stat.st_size, stat.st_mtime_ns]:
                raise ValueError(f"Image cache source changed: {row.path}")

    def load(self, path):
        pixels = np.load(self.directory / path.parent.name / (path.name + ".npy"), allow_pickle=False)
        if pixels.dtype != np.uint8 or pixels.shape != (self.image_size, self.image_size, 3):
            raise ValueError(f"Invalid image cache entry: {path}")
        return Image.fromarray(pixels)
