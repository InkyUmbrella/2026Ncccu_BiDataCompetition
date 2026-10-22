"""Single-image end-to-end latency harness for the deployment path.

Times read -> decode -> crop/letterbox -> H2D+normalize -> fp16 forward ->
sigmoid/CSV row in-process (no DataLoader), checks the 100 ms and 500 MB
deployment gates, and prints a V100 -> T4 projection. A real T4 run of this
same script is the only acceptance measurement; the projection is a prior.
"""

import argparse
import importlib.metadata
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torchvision

from glaucoma.config import Config
from glaucoma.data.age import AgeStats
from glaucoma.data.dataset import load_rgb, read_samples
from glaucoma.data.transforms import build_preprocess
from glaucoma.engine.checkpoint import load_model
from glaucoma.models import build_model
from glaucoma.utils import percentile, resolve_device, save_json, seed_everything

STAGES = ("read", "decode", "preprocess", "transfer", "forward", "post")


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _summary(values):
    return {"p50": percentile(values, 50), "p95": percentile(values, 95),
            "p99": percentile(values, 99), "max": float(max(values)), "mean": float(np.mean(values))}


def _select(rows, count, largest, rng):
    ordered = sorted(rows, key=lambda row: (-row.path.stat().st_size, row.name))
    chosen = {row.path: row for row in ordered[:largest]}
    remainder = [row for row in ordered if row.path not in chosen]
    rng.shuffle(remainder)
    for row in remainder[:max(0, count - len(chosen))]:
        chosen[row.path] = row
    return list(chosen.values())


def _pipeline(model, sample, device, dtype, decode, size, preprocess, age):
    times = {}
    started = time.perf_counter()
    with sample.path.open("rb") as handle:
        handle.read()
    times["read"] = time.perf_counter() - started

    started = time.perf_counter()
    image = load_rgb(sample.path, decode, size)
    times["decode"] = time.perf_counter() - started

    started = time.perf_counter()
    pixels = np.asarray(preprocess(image))
    image.close()
    times["preprocess"] = time.perf_counter() - started

    _sync(device)
    started = time.perf_counter()
    tensor = torch.from_numpy(pixels.copy()).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor.to(device=device, dtype=dtype) / 127.5 - 1.0
    _sync(device)
    times["transfer"] = time.perf_counter() - started

    _sync(device)
    started = time.perf_counter()
    with torch.inference_mode():
        logits = model(tensor, age)
    _sync(device)
    times["forward"] = time.perf_counter() - started

    started = time.perf_counter()
    probability = float(logits[0].sigmoid().item())
    row = f"{sample.name},{probability:.10g},{int(probability >= 0.5)}"
    times["post"] = time.perf_counter() - started
    times = {name: value * 1000 for name, value in times.items()}
    times["total"] = sum(times.values())
    return times, row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", help="best.pt; if omitted a random model times the forward only")
    parser.add_argument("--config", default="configs/experiments/b0_crop384_control.json",
                        help="Model/crop config used without --checkpoint")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    parser.add_argument("--images", type=int, default=200)
    parser.add_argument("--include-largest", type=int, default=10)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--decode", choices=("full", "draft"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu-only", action="store_true", help="CPU, fp32, gates disabled")
    parser.add_argument("--precision", choices=("fp16", "fp32"))
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--max-latency-ms", type=float, default=100.0)
    parser.add_argument("--max-model-mb", type=float, default=500.0)
    parser.add_argument("--output", help="JSON report path")
    parser.add_argument("--enforce", dest="enforce", action="store_true", default=None)
    parser.add_argument("--no-enforce", dest="enforce", action="store_false")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.images < 1 or args.include_largest < 0 or args.warmup < 0:
        raise ValueError("images must be positive and include-largest/warmup nonnegative")
    seed_everything(args.seed)
    device = resolve_device("cpu" if args.cpu_only else
                            ("cuda:0" if args.device == "cuda" else args.device))
    precision = args.precision or ("fp32" if device.type == "cpu" else "fp16")
    if device.type == "cpu" and precision == "fp16":
        raise ValueError("fp16 is not supported on CPU; use --precision fp32 or --cpu-only")
    enforce = args.enforce if args.enforce is not None else device.type == "cuda"

    if args.checkpoint:
        model, config, age_stats = load_model(args.checkpoint, device)
        artifact = Path(args.checkpoint)
        artifacts = {str(artifact): artifact.stat().st_size}
    else:
        config = Config.load(args.config)
        if args.image_size:
            config.data.image_size = args.image_size
        config.validate()
        model = build_model(config.model).to(device).eval()
        age_stats = AgeStats(0.0, 0.0, 1.0)
        artifacts = {}
    size = args.image_size or config.data.image_size
    decode = args.decode or config.data.decode
    if args.decode and args.decode != config.data.decode:
        print(f"[latency] WARNING: --decode {args.decode} overrides config decode {config.data.decode}; "
              "this is a measurement override, not a deployment setting.")
    model.eval()
    dtype = torch.float16 if precision == "fp16" else torch.float32
    if precision == "fp16":
        model.half()
    else:
        model.float()
    preprocess = build_preprocess(size, config.data.crop_black_border, config.data.crop_threshold,
                                  config.data.crop_margin)

    rows = read_samples(args.data_root, args.split, require_labels=args.split != "test")
    rng = random.Random(args.seed)
    chosen = _select(rows, args.images, args.include_largest, rng)
    params = sum(p.numel() for p in model.parameters())
    artifact_mb = sum(artifacts.values()) / 1024 ** 2

    # One timed cold run before warmup: CUDA context, cuDNN selection, first call.
    cold, _ = _pipeline(model, chosen[0], device, dtype, decode, size, preprocess,
                        torch.tensor([age_stats.transform(chosen[0].age)], device=device, dtype=dtype))
    for index in range(args.warmup):
        sample = chosen[index % len(chosen)]
        _pipeline(model, sample, device, dtype, decode, size, preprocess,
                  torch.tensor([age_stats.transform(sample.age)], device=device, dtype=dtype))

    timings = {name: [] for name in STAGES + ("total",)}
    records = []
    for sample in chosen:
        age = torch.tensor([age_stats.transform(sample.age)], device=device, dtype=dtype)
        record, row = _pipeline(model, sample, device, dtype, decode, size, preprocess, age)
        for name in timings:
            timings[name].append(record[name])
        records.append({"name": sample.name, "bytes": sample.path.stat().st_size,
                        "latency_ms": record, "row": row})
    summary = {name: _summary(values) for name, values in timings.items()}

    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
    else:
        gpu_name = "CPU"
    projection = None
    if device.type == "cuda" and "T4" not in gpu_name.upper():
        # Only the forward is GPU-bound; the CPU stages are host-specific and are
        # kept unchanged (the docs note a possible +/-50 % judge-CPU difference).
        projected = _summary([record["latency_ms"]["total"] - record["latency_ms"]["forward"]
                              + 2.0 * record["latency_ms"]["forward"] for record in records])
        projection = {"factor": 2.0, "cpu_factor": 1.0,
                      "total_p50": projected["p50"], "total_p95": projected["p95"],
                      "forward_p50": summary["forward"]["p50"] * 2.0,
                      "note": "V100->T4 batch-1 prior (1.7-2.2x) applied to forward only; "
                              "decode/preprocess are CPU-bound and not measured on the judge host"}

    latency_pass = summary["total"]["p95"] <= args.max_latency_ms
    size_pass = artifact_mb <= args.max_model_mb if artifacts else None

    print(f"GPU: {gpu_name}, torch {torch.__version__}, torchvision {torchvision.__version__}, {precision}")
    print(f"model: {args.checkpoint or args.config}, params {params / 1e6:.1f} M, "
          f"artifact {artifact_mb:.1f} MB, input {size}x{size}, decode {decode}")
    print(f"images: {len(chosen)} ({args.split}), largest included: "
          f"{max(row.path.stat().st_size for row in chosen) / 1024:.0f} KiB, warmup {args.warmup}")
    print(f"cold start (first end-to-end image): {cold['total']:.1f} ms")
    print(f"\n{'stage':<12}{'p50':>9}{'p95':>9}{'p99':>9}{'max':>9}  (ms)")
    for name in STAGES + ("total",):
        values = summary[name]
        print(f"{name:<12}{values['p50']:9.2f}{values['p95']:9.2f}{values['p99']:9.2f}{values['max']:9.2f}")
    if projection:
        print(f"\nprojected T4 (2.0x forward, CPU stages unchanged, estimate): total p50 "
              f"{projection['total_p50']:.1f} ms, p95 {projection['total_p95']:.1f} ms; "
              f"forward p50 {projection['forward_p50']:.1f} ms")
    print(f"GATE latency p95 <= {args.max_latency_ms:.0f} ms: {'PASS' if latency_pass else 'FAIL'} "
          f"({summary['total']['p95']:.1f} ms)")
    if size_pass is None:
        print(f"GATE model artifact <= {args.max_model_mb:.0f} MB: SKIP (no checkpoint)")
    else:
        print(f"GATE model artifact <= {args.max_model_mb:.0f} MB: {'PASS' if size_pass else 'FAIL'} "
              f"({artifact_mb:.1f} MB)")
    if args.output:
        save_json(args.output, {
            "gpu": gpu_name, "device": str(device), "precision": precision, "decode": decode,
            "checkpoint": str(args.checkpoint) if args.checkpoint else None,
            "artifacts": {name: {"bytes": size} for name, size in artifacts.items()},
            "params": params, "image_size": size, "split": args.split, "warmup": args.warmup,
            "cold_start_ms": cold["total"], "summary_ms": summary, "records": records,
            "projection_t4": projection,
            "gates": {"latency_p95_le": args.max_latency_ms, "latency_pass": latency_pass,
                      "model_mb_le": args.max_model_mb,
                      "model_size_pass": size_pass, "enforced": bool(enforce)},
            "versions": {"torch": torch.__version__, "torchvision": torchvision.__version__,
                         "numpy": np.__version__,
                         "Pillow": importlib.metadata.version("Pillow"),
                         "python": sys.version.split()[0]},
        })
        print(f"report: {args.output}")
    failed = not latency_pass or size_pass is False
    if enforce and failed:
        raise SystemExit(1)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
