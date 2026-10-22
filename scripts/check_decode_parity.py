"""Compare full and draft JPEG decoding: timing, tensor parity and prediction parity.

Read-only analysis: writes nothing except the optional ``--output`` JSON. The
draft policy is accepted only if the tensor and (when a checkpoint is given)
AUROC gates pass on real images.
"""

import argparse
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image

from glaucoma.config import Config
from glaucoma.data.age import AgeStats
from glaucoma.data.dataset import FundusDataset, load_rgb, read_samples
from glaucoma.data.transforms import CropBlackBorder, build_preprocess, build_transform
from glaucoma.engine.checkpoint import read_checkpoint
from glaucoma.engine.evaluation import collect_predictions
from glaucoma.engine.trainer import make_loader
from glaucoma.metrics import binary_metrics
from glaucoma.models import build_model
from glaucoma.utils import percentile, resolve_device, save_json, seed_everything

TENSOR_MEAN_P95_GATE = 0.010
BBOX_FRACTION_DRIFT_GATE = 0.01
AUROC_DELTA_GATE = 0.001
PREPROCESSING_FIELDS = ("image_size", "crop_black_border", "crop_threshold",
                        "crop_margin", "decode")


def sample_rows(rows, count, largest, rng):
    """Deterministic subset, always including the largest files by byte size."""
    ordered = sorted(rows, key=lambda row: (-row.path.stat().st_size, row.name))
    if count == 0:
        return ordered, ordered[:largest]
    chosen = {row.path: row for row in ordered[:largest]}
    remainder = [row for row in ordered if row.path not in chosen]
    if count > 0 and len(chosen) < count:
        rng.shuffle(remainder)
        for row in remainder[:count - len(chosen)]:
            chosen[row.path] = row
    return sorted(chosen.values(), key=lambda row: row.name), ordered[:largest]


def tensor_for(image, transform):
    return transform(image)


def _image_size(path):
    with Image.open(path) as image:
        return image.size


def bbox_fraction(image, crop):
    box = crop.bbox(image)
    if box is None:
        return None
    left, top, right, bottom = box
    return [left / image.width, top / image.height, right / image.width, bottom / image.height]


def timing(values):
    return {"count": len(values), "p50": percentile(values, 50), "p95": percentile(values, 95),
            "p99": percentile(values, 99), "max": float(np.max(values)) if len(values) else None}


def decode_timing(paths, size, crop_args, repeats):
    stages = {"read": [], "full_decode": [], "draft_decode": [],
              "full_preprocess": [], "draft_preprocess": []}
    for path in paths:
        read_times, full_decode, draft_decode, full_pp, draft_pp = [], [], [], [], []
        for _ in range(repeats):
            started = time.perf_counter()
            with path.open("rb") as handle:
                handle.read()
            read_times.append((time.perf_counter() - started) * 1000)
            started = time.perf_counter()
            full = load_rgb(path, "full", size)
            full_decode.append((time.perf_counter() - started) * 1000)
            started = time.perf_counter()
            draft = load_rgb(path, "draft", size)
            draft_decode.append((time.perf_counter() - started) * 1000)
            started = time.perf_counter()
            build_preprocess(size, **crop_args)(full)
            full_pp.append((time.perf_counter() - started) * 1000)
            started = time.perf_counter()
            build_preprocess(size, **crop_args)(draft)
            draft_pp.append((time.perf_counter() - started) * 1000)
            full.close()
            draft.close()
        stages["read"].append(float(np.median(read_times)))
        stages["full_decode"].append(float(np.median(full_decode)))
        stages["draft_decode"].append(float(np.median(draft_decode)))
        stages["full_preprocess"].append(float(np.median(full_pp)))
        stages["draft_preprocess"].append(float(np.median(draft_pp)))
    return {name: timing(values) for name, values in stages.items()}


def tensor_parity(rows, size, crop_args):
    transform = build_transform(size, preprocessed=True)
    crop = CropBlackBorder(crop_args["crop_threshold"], crop_args["crop_margin"]) \
        if crop_args["crop_black_border"] else None
    full_preprocess = build_preprocess(size, **crop_args)
    rows_out = []
    means, maxima, elementwise_p99, elementwise_max = [], [], [], []
    bbox_deltas, bbox_disagreements, bbox_both_unavailable, scales = [], 0, 0, []
    for row in rows:
        full_image = load_rgb(row.path, "full", size)
        draft_image = load_rgb(row.path, "draft", size)
        full_tensor = tensor_for(full_preprocess(full_image), transform)
        draft_tensor = tensor_for(full_preprocess(draft_image), transform)
        delta = (full_tensor - draft_tensor).abs().numpy().ravel()
        mean_abs, max_abs = float(delta.mean()), float(delta.max())
        q99 = float(np.percentile(delta, 99))
        elementwise_p99.append(q99)
        elementwise_max.append(max_abs)
        means.append(mean_abs)
        maxima.append(max_abs)
        if crop is not None:
            left, right = bbox_fraction(full_image, crop), bbox_fraction(draft_image, crop)
            if left is None and right is None:
                bbox_both_unavailable += 1
            elif left is None or right is None:
                # One decode finds a field and the other does not: a real framing
                # disagreement (the dominant deployment risk), never a small delta.
                bbox_disagreements += 1
            else:
                bbox_deltas.append(float(np.max(np.abs(np.asarray(left) - np.asarray(right)))))
        scales.append(round(max(full_image.size) / max(draft_image.size), 3))
        rows_out.append({"name": row.name, "source_size": list(full_image.size),
                         "draft_size": list(draft_image.size),
                         "mean_abs": mean_abs, "max_abs": max_abs, "elementwise_p99": q99})
        full_image.close()
        draft_image.close()
    bbox_max_delta = float(max(bbox_deltas)) if bbox_deltas else None
    bbox_gate = crop is None or (bbox_disagreements == 0
                                 and (bbox_max_delta is None
                                      or bbox_max_delta <= BBOX_FRACTION_DRIFT_GATE))
    tensor_gate = timing(means)["p95"] <= TENSOR_MEAN_P95_GATE
    result = {
        "count": len(rows),
        "mean_abs": timing(means),
        "max_abs": float(max(maxima)) if maxima else None,
        "elementwise_p99_across_images": float(max(elementwise_p99)) if elementwise_p99 else None,
        "elementwise_max": float(max(elementwise_max)) if elementwise_max else None,
        "bbox_fraction_max_delta": bbox_max_delta,
        "bbox_disagreements": bbox_disagreements,
        "bbox_both_unavailable": bbox_both_unavailable,
        "draft_scale_factor": timing(scales),
        "per_image": rows_out,
        "tensor_gate_pass": tensor_gate,
        "bbox_gate_pass": bbox_gate,
    }
    result["gate_pass"] = tensor_gate and bbox_gate
    return result


def prediction_parity(rows, payload, checkpoint, batch_size, seed, device, workers):
    """Run the checkpoint on the complete split with both decode policies."""
    checkpoint_config = Config.from_dict(payload["config"])
    size = checkpoint_config.data.image_size
    crop_args = {"crop_black_border": checkpoint_config.data.crop_black_border,
                 "crop_threshold": checkpoint_config.data.crop_threshold,
                 "crop_margin": checkpoint_config.data.crop_margin}
    model = build_model(checkpoint_config.model)
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device).eval()
    age_stats = AgeStats(**payload["age_stats"])
    results = {}
    for decode in ("full", "draft"):
        dataset = FundusDataset(rows, age_stats,
                                build_transform(size, preprocessed=False, **crop_args),
                                decode=decode, image_size=size)
        loader = make_loader(dataset, batch_size, workers, seed,
                             pin_memory=device.type == "cuda")
        predictions, metrics = collect_predictions(model, loader, device,
                                                   checkpoint_config.train.pos_weight, amp=False)
        results[decode] = (predictions, metrics)
    full, draft = results["full"], results["draft"]
    if full[0]["names"] != draft[0]["names"]:
        raise RuntimeError("Decode passes returned different sample order")
    p_full = np.asarray(full[0]["probabilities"], dtype=np.float64)
    p_draft = np.asarray(draft[0]["probabilities"], dtype=np.float64)
    delta = np.abs(p_full - p_draft)
    from scipy.stats import spearmanr
    auroc_full = full[1]["auroc"]
    auroc_draft = draft[1]["auroc"]
    auroc_delta = abs(auroc_full - auroc_draft)
    result = {
        "count": len(p_full), "checkpoint": str(checkpoint), "checkpoint_decode": checkpoint_config.data.decode,
        "auroc_full": auroc_full, "auroc_draft": auroc_draft, "auroc_delta": auroc_delta,
        "probability_delta": timing(delta),
        "spearman": float(spearmanr(p_full, p_draft).statistic),
        "gate_pass": auroc_delta <= AUROC_DELTA_GATE,
    }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", choices=("train", "valid"), default="valid")
    parser.add_argument("--samples", type=int, default=200, help="Tensor-parity images; 0 = all")
    parser.add_argument("--include-largest", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--config", help="Experiment config; supplies size, crop and decode defaults")
    parser.add_argument("--checkpoint", help="best.pt; enables prediction parity on the full split")
    parser.add_argument("--decode-compare", default="draft", choices=("draft",),
                        help="Policy compared against full")
    parser.add_argument("--batch-size", type=int, default=16, help="Checkpoint pass batch size")
    parser.add_argument("--workers", type=int, default=4, help="Checkpoint pass data-loader workers")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--timing", action="store_true", help="Also run the decode micro-benchmark")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", help="JSON report path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-enforce", dest="enforce", action="store_false", default=True)
    args = parser.parse_args(argv)
    if args.samples < 0 or args.include_largest < 0 or args.repeats < 1:
        raise ValueError("samples/include-largest must be nonnegative and repeats positive")
    seed_everything(args.seed)
    supplied = Config.load(args.config) if args.config else None
    config = supplied or Config()
    payload = read_checkpoint(args.checkpoint, "inference") if args.checkpoint else None
    if payload is not None:
        # The checkpoint owns preprocessing: measuring a different crop/size than
        # the deployed model is a false pass, so conflicting overrides are errors.
        checkpoint_config = Config.from_dict(payload["config"])
        conflicts = []
        if supplied is not None:
            for name in PREPROCESSING_FIELDS:
                wanted, actual = getattr(supplied.data, name), getattr(checkpoint_config.data, name)
                if wanted != actual:
                    conflicts.append(f"data.{name}: --config={wanted!r} checkpoint={actual!r}")
        if args.image_size is not None and args.image_size != checkpoint_config.data.image_size:
            conflicts.append(f"data.image_size: --image-size={args.image_size} "
                             f"checkpoint={checkpoint_config.data.image_size}")
        if conflicts:
            raise SystemExit("--config/--image-size conflict with the checkpoint's preprocessing: "
                             + "; ".join(conflicts))
        size = checkpoint_config.data.image_size
        crop_args = {"crop_black_border": checkpoint_config.data.crop_black_border,
                     "crop_threshold": checkpoint_config.data.crop_threshold,
                     "crop_margin": checkpoint_config.data.crop_margin}
        config_source = "checkpoint"
    else:
        size = args.image_size or config.data.image_size
        crop_args = {"crop_black_border": config.data.crop_black_border,
                     "crop_threshold": config.data.crop_threshold,
                     "crop_margin": config.data.crop_margin}
        config_source = "config" if supplied is not None else "defaults"
    rows = read_samples(args.data_root, args.split)
    rng = random.Random(args.seed)
    chosen, largest = sample_rows(rows, args.samples, args.include_largest, rng)
    report = {"data_root": str(Path(args.data_root).resolve()), "split": args.split,
              "image_size": size, "crop": crop_args, "config_source": config_source,
              "config": str(args.config) if args.config else None,
              "checkpoint": str(args.checkpoint) if args.checkpoint else None,
              "compare": ["full", args.decode_compare],
              "sampled": [{"name": row.name, "bytes": row.path.stat().st_size,
                           "size": list(_image_size(row.path))} for row in chosen],
              "largest": [row.name for row in largest]}
    print(f"[parity] tensor comparison on {len(chosen)}/{len(rows)} {args.split} images at {size}px "
          f"(preprocessing from {config_source})")
    report["tensor"] = tensor_parity(chosen, size, crop_args)
    tensor_gate = report["tensor"]["tensor_gate_pass"]
    bbox_gate = report["tensor"]["bbox_gate_pass"]
    print(f"  mean |delta| p50={report['tensor']['mean_abs']['p50']:.5f} "
          f"p95={report['tensor']['mean_abs']['p95']:.5f} max={report['tensor']['max_abs']:.5f}")
    print(f"  bbox fraction max delta={report['tensor']['bbox_fraction_max_delta']} "
          f"disagreements={report['tensor']['bbox_disagreements']} "
          f"both-unavailable={report['tensor']['bbox_both_unavailable']}")
    print(f"  GATE tensor mean |delta| p95 <= {TENSOR_MEAN_P95_GATE}: "
          f"{'PASS' if tensor_gate else 'FAIL'}")
    print(f"  GATE bbox max drift <= {BBOX_FRACTION_DRIFT_GATE} and zero None/crop "
          f"disagreements: {'PASS' if bbox_gate else 'FAIL'}")
    if args.timing:
        paths = [row.path for row in chosen]
        report["timing_ms"] = decode_timing(paths, size, crop_args, args.repeats)
        print("[parity] decode timing (ms, per-image median of repeats):")
        for name, values in report["timing_ms"].items():
            print(f"  {name:18s} p50={values['p50']:8.2f} p95={values['p95']:8.2f}")
    checkpoint_gate = None
    if args.checkpoint:
        device = resolve_device("cuda:0" if args.device == "cuda" else args.device)
        print(f"[parity] checkpoint pass on the complete {args.split} split ({len(rows)} images, {device})")
        report["prediction"] = prediction_parity(rows, payload, args.checkpoint, args.batch_size,
                                                 config.seed, device, args.workers)
        prediction = report["prediction"]
        checkpoint_gate = prediction["gate_pass"]
        print(f"  AUROC full={prediction['auroc_full']:.6f} draft={prediction['auroc_draft']:.6f} "
              f"|delta|={prediction['auroc_delta']:.6f}")
        print(f"  probability |delta| p50={prediction['probability_delta']['p50']:.5f} "
              f"p99={prediction['probability_delta']['p99']:.5f} max={prediction['probability_delta']['max']:.5f} "
              f"spearman={prediction['spearman']:.6f}")
        print(f"  GATE AUROC |delta| <= {AUROC_DELTA_GATE}: {'PASS' if checkpoint_gate else 'FAIL'}")
    if args.output:
        save_json(args.output, report)
        print(f"[parity] report: {args.output}")
    passed = tensor_gate and bbox_gate and checkpoint_gate is not False
    if args.enforce and not passed:
        raise SystemExit(1)
    print(f"[parity] {'PASS' if passed else 'FAIL'} "
          f"(tensor={'PASS' if tensor_gate else 'FAIL'}, "
          f"bbox={'PASS' if bbox_gate else 'FAIL'}, "
          f"auroc={'n/a' if checkpoint_gate is None else ('PASS' if checkpoint_gate else 'FAIL')})")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
