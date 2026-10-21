"""Entrypoints stay thin; model/data/engine code is reusable without the CLI."""

import argparse
from pathlib import Path

from .config import Config


def build_parser():
    parser = argparse.ArgumentParser(description="Glaucoma baseline training and inference")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("models", help="List registered models")
    cache = commands.add_parser("cache-images", help="Build lossless train/valid preprocessing cache")
    cache.add_argument("--config", help="Use the experiment's size and preprocessing settings")
    cache.add_argument("--data-root")
    cache.add_argument("--output", required=True, help="New cache directory, preferably on local disk")
    cache.add_argument("--image-size", type=int)
    cache.add_argument("--workers", type=int, default=4)
    cache.add_argument("--crop-black-border", action=argparse.BooleanOptionalAction, default=None)
    cache.add_argument("--crop-threshold", type=int)
    cache.add_argument("--crop-margin", type=float)
    audit = commands.add_parser("check-data", help="Decode images and optionally check exact duplicates")
    audit.add_argument("--data-root", required=True)
    audit.add_argument("--splits", nargs="+", choices=("train", "valid", "test"), default=["train", "valid", "test"])
    audit.add_argument("--hashes", action="store_true", help="Also hash bytes; may be slow on NFS")
    audit.add_argument("--output", required=True, help="New JSON report path")
    training = commands.add_parser("train")
    training.add_argument("--config", required=True)
    training.add_argument("--data-root")
    training.add_argument("--device")
    training.add_argument("--output-root")
    initialization = training.add_mutually_exclusive_group()
    initialization.add_argument("--resume", help="last.pt; starts a new output directory")
    initialization.add_argument("--init-from", help="best.pt; initialize a new experiment without optimizer/history")
    training.add_argument("--seed", type=int)
    training.add_argument("--image-cache", help="Lossless cache built by cache-images; resume compatible")
    training.add_argument("--stop-after-epochs", type=int, help="Bound this invocation without changing the LR schedule")
    for name in ("evaluate", "predict"):
        command = commands.add_parser(name)
        command.add_argument("--checkpoint", required=True, help="best.pt inference checkpoint")
        command.add_argument("--data-root", required=True)
        command.add_argument("--device", default="cuda")
        command.add_argument("--batch-size", type=int, default=16)
        command.add_argument("--num-workers", type=int, default=4)
        if name == "evaluate":
            command.add_argument("--split", choices=("train", "valid"), default="valid")
            command.add_argument("--output-dir", required=True, help="New directory")
        else:
            command.add_argument("--output", required=True, help="New submission CSV path")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "models":
        from .models import available_models
        print("\n".join(available_models()))
    elif args.command == "cache-images":
        from .data.cache import prepare_cache
        config = Config.load(args.config) if args.config else Config()
        if not args.config and not args.data_root:
            raise ValueError("cache-images requires --config or --data-root")
        for name in ("image_size", "crop_black_border", "crop_threshold", "crop_margin"):
            value = getattr(args, name)
            if value is not None:
                setattr(config.data, name, value)
        config.validate()
        prepare_cache(args.data_root or config.data.root, args.output, config.data.image_size, args.workers,
                      crop_black_border=config.data.crop_black_border,
                      crop_threshold=config.data.crop_threshold, crop_margin=config.data.crop_margin)
    elif args.command == "train":
        from .engine.trainer import train
        config = Config.load(args.config)
        if args.data_root:
            config.data.root = args.data_root
        if args.device:
            config.device = args.device
        if args.output_root:
            config.output_root = args.output_root
        if args.seed is not None:
            config.seed = args.seed
        run = train(config, args.resume, args.stop_after_epochs, image_cache=args.image_cache,
                    init_from=args.init_from)
        print(f"Results: {run.resolve()}")
    elif args.command == "check-data":
        from .data.audit import audit_samples
        from .data.dataset import read_samples
        from .utils import save_json
        if Path(args.output).exists():
            raise FileExistsError(args.output)
        splits = {split: read_samples(args.data_root, split, split != "test") for split in args.splits}
        report = audit_samples(splits, args.hashes)
        save_json(args.output, report)
        print(f"Audit: {args.output}; errors={len(report['errors'])}; duplicate_groups={len(report['duplicate_groups'])}")
        if report["errors"]:
            raise SystemExit(1)
    else:
        _inference(args)


def _inference(args):
    from .data.dataset import FundusDataset, read_samples
    from .data.transforms import build_transform
    from .engine.checkpoint import load_model
    from .engine.evaluation import collect_predictions, write_predictions
    from .engine.trainer import make_loader
    from .utils import resolve_device, save_json
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers nonnegative")
    predicting = args.command == "predict"
    destination = Path(args.output if predicting else args.output_dir)
    if destination.exists():
        raise FileExistsError(destination)
    device = resolve_device(args.device)
    model, config, age_stats = load_model(args.checkpoint, device)
    rows = read_samples(args.data_root, "test" if predicting else args.split, not predicting)
    dataset = FundusDataset(rows, age_stats, build_transform(config.data.image_size,
                           crop_black_border=config.data.crop_black_border,
                           crop_threshold=config.data.crop_threshold, crop_margin=config.data.crop_margin))
    loader = make_loader(dataset, args.batch_size, args.num_workers, config.seed, pin_memory=device.type == "cuda")
    predictions, metrics = collect_predictions(model, loader, device, config.train.pos_weight, config.train.amp)
    if predicting:
        write_predictions(destination, predictions, submission=True)
    else:
        destination.mkdir(parents=True, exist_ok=False)
        write_predictions(destination / "predictions.csv", predictions)
        save_json(destination / "metrics.json", metrics)
    print(f"Wrote {len(rows)} predictions to {destination.resolve()}")
