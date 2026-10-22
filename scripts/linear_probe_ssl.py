#!/usr/bin/env python3
"""Frozen linear-probe gate for SSL init vs random init (both at 384).

Run on the GPU server:

    CUDA_VISIBLE_DEVICES=5 PYTHONPATH=src python scripts/linear_probe_ssl.py \
        --encoder outputs/ssl/.../encoder_last.pt \
        --config configs/ssl/moco_v2_224_200ep.json \
        --output outputs/ssl/probe_gate.json

The probe is fitted on the train split and evaluated on valid. The SSL loss
never saw a label; labels are used only by this supervised read-out.
"""

import argparse
import json
from pathlib import Path
import random
import sys

import torch
from torch.utils.data import DataLoader

from glaucoma.data.age import AgeStats
from glaucoma.data.cache import ImageCache
from glaucoma.data.dataset import FundusDataset, read_samples
from glaucoma.engine.checkpoint import read_ssl_encoder
from glaucoma.ssl.augment import build_probe_transform
from glaucoma.ssl.config import SSLConfig
from glaucoma.ssl.model import build_encoder
from glaucoma.ssl.probe import effective_rank, extract_features, feature_std, probe_metrics
from glaucoma.utils import save_json, seed_everything, sha256_file


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", required=True, help="kind=ssl_encoder .pt from pretraining")
    parser.add_argument("--config", help="SSL JSON config; supplies data root/cache/preprocessing")
    parser.add_argument("--data-root")
    parser.add_argument("--image-cache")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--output", required=True, help="New JSON report path")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-train", type=int, default=None, help="Optional deterministic subsample")
    parser.add_argument("--max-valid", type=int, default=None)
    parser.add_argument("--C", type=float, default=1.0, help="Inverse logistic-regression regularisation")
    return parser.parse_args(argv)


def subset(rows, limit, seed):
    if limit is None or limit >= len(rows):
        return list(rows)
    indices = sorted(random.Random(seed).sample(range(len(rows)), limit))
    return [rows[i] for i in indices]


def main(argv=None):
    args = parse_args(argv)
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers nonnegative")
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError(destination)
    payload = read_ssl_encoder(args.encoder)
    config = SSLConfig.load(args.config) if args.config else SSLConfig()
    if args.data_root:
        config.data.root = args.data_root
    if args.image_cache:
        config.data.cache = args.image_cache
    if args.image_size != config.data.image_size:
        config.data.image_size = args.image_size
    config.validate()
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    train_rows = subset(read_samples(config.data.root, "train", require_labels=True),
                        args.max_train, args.seed)
    valid_rows = subset(read_samples(config.data.root, "valid", require_labels=True),
                        args.max_valid, args.seed + 1)
    for split, rows in (("train", train_rows), ("valid", valid_rows)):
        if {row.label for row in rows} != {0, 1}:
            raise ValueError(f"{split} probe subset must contain both classes")
    age_stats = AgeStats.fit(row.age for row in read_samples(config.data.root, "train"))
    cache = None
    if config.data.cache:
        cache = ImageCache(config.data.cache, read_samples(config.data.root, "train"),
                           config.data.image_size, crop_black_border=config.data.crop_black_border,
                           crop_threshold=config.data.crop_threshold,
                           crop_margin=config.data.crop_margin, decode=config.data.decode)
    transform = build_probe_transform(
        config.data.image_size, crop_black_border=config.data.crop_black_border,
        crop_threshold=config.data.crop_threshold, crop_margin=config.data.crop_margin,
        preprocessed=cache is not None)

    def loaders(rows):
        dataset = FundusDataset(rows, age_stats, transform, image_cache=cache,
                                image_size=config.data.image_size)
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                          num_workers=args.num_workers, pin_memory=device.type == "cuda")

    def features(encoder):
        encoder.to(device).eval()
        train_x, train_y, _ = extract_features(encoder, loaders(train_rows), device, config.train.amp)
        valid_x, valid_y, _ = extract_features(encoder, loaders(valid_rows), device, config.train.amp)
        return train_x, train_y, valid_x, valid_y

    ssl_encoder = build_encoder()
    ssl_encoder.load_state_dict(payload["encoder_state"], strict=True)
    ssl_features = features(ssl_encoder)

    torch.manual_seed(args.seed)
    random_encoder = build_encoder()
    random_features = features(random_encoder)

    report = {"encoder": str(Path(args.encoder).resolve()),
              "encoder_sha256": sha256_file(args.encoder),
              "ssl_epoch": payload.get("epoch"),
              "protocol": {"image_size": config.data.image_size, "seed": args.seed,
                           "probe": "StandardScaler+LogisticRegression(class_weight=balanced)",
                           "train_count": len(train_rows), "valid_count": len(valid_rows),
                           "C": args.C, "amp": config.train.amp}}
    for name, (train_x, train_y, valid_x, valid_y) in (
            ("ssl", ssl_features), ("random", random_features)):
        metrics = probe_metrics(train_x, train_y, valid_x, valid_y, C=args.C)
        metrics.update(feature_dim=int(train_x.shape[1]),
                       effective_rank_valid=effective_rank(valid_x),
                       feature_std_valid=feature_std(valid_x))
        report[name] = metrics
    report["delta_auroc"] = (report["ssl"]["auroc"] - report["random"]["auroc"]
                             if report["ssl"]["auroc"] is not None
                             and report["random"]["auroc"] is not None else None)
    delta = report["delta_auroc"]
    if delta is None:
        decision = "inconclusive"
    elif delta >= 0.010:
        decision = "go"
    elif delta >= 0.005:
        decision = "candidate"
    else:
        decision = "abandon"
    report["decision"] = decision
    report["decision_rule"] = "go >= +0.010; candidate +0.005..0.010; abandon < +0.005"
    save_json(destination, report)
    print(json.dumps({key: report[key] for key in ("ssl", "random", "delta_auroc", "decision")},
                     indent=2), flush=True)
    print(f"Wrote {destination.resolve()}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
