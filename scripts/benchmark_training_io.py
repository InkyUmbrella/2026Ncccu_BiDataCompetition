"""Compare original images and prepared lossless cache on an idle allocated GPU."""

import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from glaucoma.config import Config
from glaucoma.data.age import AgeStats
from glaucoma.data.dataset import FundusDataset, read_samples
from glaucoma.data.transforms import build_transform
from glaucoma.engine.optimization import build_optimizer, train_one_epoch
from glaucoma.engine.trainer import make_loader
from glaucoma.models import build_model
from glaucoma.utils import seed_everything


class ProbeCache:
    def __init__(self, directory):
        self.directory = Path(directory)

    def load(self, path):
        return Image.fromarray(np.load(self.directory / path.parent.name / (path.name + ".npy"),
                                       allow_pickle=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = Config.load(args.config)
    all_rows = read_samples(config.data.root, "train")
    cache = ProbeCache(args.cache)
    # A cache under construction may be benchmarked using only completed files.
    available = [row for row in all_rows
                 if (cache.directory / "train" / (row.path.name + ".npy")).exists()]
    if len(available) < args.samples or args.samples < config.train.batch_size * 4:
        raise ValueError("Need at least four batches and enough completed cache files")
    indices = np.random.default_rng(42).choice(len(available), args.samples, replace=False)
    rows = [available[index] for index in indices]
    age = AgeStats.fit(row.age for row in all_rows)
    original = FundusDataset(rows, age, build_transform(config.data.image_size, True))
    cached = FundusDataset(rows, age, build_transform(config.data.image_size, True), cache)
    for index in range(min(32, len(rows))):
        torch.manual_seed(index)
        expected = original[index]["image"]
        torch.manual_seed(index)
        torch.testing.assert_close(cached[index]["image"], expected, rtol=0, atol=0)
    print("[benchmark] 32 real-image augmented tensors match exactly", flush=True)
    device = torch.device(config.device)
    if device.type != "cuda":
        raise ValueError("This benchmark requires an allocated idle CUDA GPU")
    results = {}
    for name, dataset in (("original", original), ("cached", cached)):
        seed_everything(config.seed)
        model = build_model(config.model).to(device)
        optimizer = build_optimizer(model, config.train)
        scaler = torch.amp.GradScaler("cuda", enabled=config.train.amp)
        # Warm up kernels before timing the real DataLoader and training loop.
        with torch.autocast("cuda", dtype=torch.float16, enabled=config.train.amp):
            model(torch.zeros(config.train.batch_size, 3, config.data.image_size,
                              config.data.image_size, device=device),
                  torch.zeros(config.train.batch_size, 3, device=device)).sum().backward()
        optimizer.zero_grad(set_to_none=True)
        loader = make_loader(dataset, config.train.batch_size, config.data.num_workers,
                             config.seed, training=True, pin_memory=True)
        torch.cuda.synchronize()
        started = time.perf_counter()
        metrics, _ = train_one_epoch(model, loader, optimizer, scaler, device, config.train, 0, 100, 5)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        results[name] = {"samples": len(rows), "seconds": seconds,
                         "images_per_second": len(rows) / seconds, "metrics": metrics}
        print(f"[benchmark] {name}: {seconds:.3f}s, {len(rows)/seconds:.2f} images/s", flush=True)
        del model, optimizer, scaler, loader
        gc.collect()
        torch.cuda.empty_cache()
    results["speedup"] = results["original"]["seconds"] / results["cached"]["seconds"]
    with Path(args.output).open("x", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
