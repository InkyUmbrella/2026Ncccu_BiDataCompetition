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
    source, target, size, crop, threshold, margin, decode, storage = task
    source = Path(source)
    before = source.stat()
    pixels = np.asarray(build_preprocess(size, crop, threshold, margin)(
        load_rgb(source, decode, size)))
    if storage == "png":
        atomic_write(target, lambda handle: Image.fromarray(pixels).save(handle, format="PNG", compress_level=1))
    else:
        atomic_write(target, lambda handle: np.save(handle, pixels, allow_pickle=False))
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"Source changed while caching: {source}")
    return source.parent.name + "/" + source.name, [after.st_size, after.st_mtime_ns]


def prepare_cache(data_root, output, image_size=384, workers=4, *, crop_black_border=False,
                  crop_threshold=10, crop_margin=0.03, decode="full", storage="npy"):
    if image_size < 32 or workers < 1:
        raise ValueError("image_size must be >=32 and workers must be positive")
    if storage not in ("npy", "png"):
        raise ValueError("storage must be npy or png")
    from ..config import Config
    Config.from_dict({"data": {"image_size": image_size, "crop_black_border": crop_black_border,
                              "crop_threshold": crop_threshold, "crop_margin": crop_margin,
                              "decode": decode}})
    root, output = Path(data_root).resolve(), Path(output).resolve()
    rows = [row for split in ("train", "valid") for row in read_samples(root, split)]
    output.mkdir(parents=True, exist_ok=False)
    estimate = len(rows) * (image_size * image_size * 3 + 256)
    # PNG's exact size is data-dependent; enforce a conservative per-image
    # minimum here. The launcher separately budgets sampled compressed sizes.
    required = estimate * 1.1 if storage == "npy" else len(rows) * 4096
    if shutil.disk_usage(output).free < required:
        raise ValueError(f"Cache requires approximately {estimate / 1024**3:.1f} GiB plus headroom")
    hashes = {split: sha256_file(root / f"{split}.csv") for split in ("train", "valid")}
    tasks = [(str(row.path), str(output / row.path.parent.name / (row.path.name + "." + storage)),
              image_size, crop_black_border, crop_threshold, crop_margin, decode, storage)
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
    save_json(output / "manifest.json", {"version": 4 if storage == "png" else 3,
              "storage": storage, "source_root": str(root), "decode": decode,
              "preprocessing": preprocessing_spec(crop_black_border, crop_threshold, crop_margin, decode),
              "image_size": image_size, "csv_hashes": hashes, "entries": entries})
    print(f"[cache] complete: {output}", flush=True)


class ImageCache:
    def __init__(self, directory, samples, image_size, *, crop_black_border=False,
                 crop_threshold=10, crop_margin=0.03, decode="full"):
        self.directory = Path(directory)
        manifest = json.loads((self.directory / "manifest.json").read_text(encoding="utf-8"))
        version = manifest["version"]
        if version not in (1, 2, 3, 4) or manifest["image_size"] != image_size:
            raise ValueError("Image cache version or image size mismatch")
        self.storage = manifest.get("storage", "npy")
        if self.storage not in ("npy", "png") or (version < 4 and self.storage != "npy"):
            raise ValueError("Unsupported image cache storage")
        # Version 1/2 manifests predate the decode policy and can only be full decodes.
        stored = {"name": "letterbox_v1"} if version == 1 else dict(manifest["preprocessing"])
        stored.setdefault("decode", "full")
        if version >= 3 and manifest.get("decode", stored["decode"]) != stored["decode"]:
            raise ValueError("Image cache manifest decode fields disagree")
        if stored["decode"] != decode:
            raise ValueError("Image cache decode mismatch; rebuild from original images")
        if stored != preprocessing_spec(crop_black_border, crop_threshold, crop_margin, decode):
            raise ValueError("Image cache preprocessing mismatch; rebuild from original images")
        self.image_size = image_size
        roots = {root.resolve() for root in {row.path.parent.parent for row in samples}}
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
        if self.storage == "png":
            with Image.open(self.directory / path.parent.name / (path.name + ".png")) as image:
                image.load()
                if image.mode != "RGB" or image.size != (self.image_size, self.image_size):
                    raise ValueError(f"Invalid image cache entry: {path}")
                return image.copy()
        pixels = np.load(self.directory / path.parent.name / (path.name + ".npy"), allow_pickle=False)
        if pixels.dtype != np.uint8 or pixels.shape != (self.image_size, self.image_size, 3):
            raise ValueError(f"Invalid image cache entry: {path}")
        return Image.fromarray(pixels)
