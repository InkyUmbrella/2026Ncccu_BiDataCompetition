"""Frozen batch-one prediction shared by predict.py, cost.py and the CLI."""

import csv
import hashlib
import json
import math
import os
from pathlib import Path

import torch

from .config import Config
from .data.dataset import load_rgb
from .data.transforms import build_transform
from .engine.checkpoint import load_model, read_checkpoint
from .utils import resolve_device, seed_everything, sha256_file


SUBMISSION_FIELDS = ["Image_name", "Probability", "Pred_Label"]
DEPLOYMENT_LIMIT = 500_000_000


def source_hashes():
    """Normalize newlines so a frozen package can move between Windows/Linux."""
    package = Path(__file__).resolve().parent
    repository = package.parents[1]
    paths = {"src/glaucoma/" + path.relative_to(package).as_posix(): path
             for path in package.rglob("*.py")}
    for name in ("predict.py", "cost.py", "scripts/freeze_inference.py"):
        path = repository / name
        if not path.is_file():
            raise FileNotFoundError(f"Inference entrypoint missing: {path}")
        paths[name] = path
    return {name: hashlib.sha256(path.read_text(encoding="utf-8").replace("\r\n", "\n")
                                .encode("utf-8")).hexdigest() for name, path in sorted(paths.items())}


def preprocess_spec(config):
    return {key: getattr(config.data, key) for key in (
        "image_size", "normalization", "crop_black_border", "crop_threshold", "crop_margin")}


def write_json_exclusive(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def freeze_spec(checkpoint, destination, precision="fp16", tta="none"):
    if precision not in ("fp16", "fp32") or tta not in ("none", "hflip"):
        raise ValueError("Supported precision: fp16/fp32; TTA: none/hflip")
    checkpoint, destination = Path(checkpoint).resolve(), Path(destination).resolve()
    if destination.exists() or checkpoint.stat().st_size >= DEPLOYMENT_LIMIT:
        raise ValueError("Spec must be new and checkpoint must leave room under 500MB")
    digest = sha256_file(checkpoint)
    payload = read_checkpoint(checkpoint, "inference")
    config = Config.from_dict(payload["config"])
    if not payload.get("model_state") or any(not isinstance(value, torch.Tensor)
            or not torch.isfinite(value).all() for value in payload["model_state"].values()):
        raise ValueError("Inference checkpoint must contain finite model tensors")
    if sha256_file(checkpoint) != digest:
        raise ValueError("Checkpoint changed during spec creation")
    try:
        reference = os.path.relpath(checkpoint, destination.parent)
    except ValueError:  # Different Windows drives; re-freeze after packaging for portability.
        reference = str(checkpoint)
    spec = {"format_version": 1, "kind": "glaucoma_single_image", "precision": precision,
            "tta": tta, "threshold": 0.5,
            "checkpoint": {"file": Path(reference).as_posix(), "sha256": digest,
                           "bytes": checkpoint.stat().st_size},
            "model": config.model.__dict__, "preprocessing": preprocess_spec(config),
            "age_stats": payload["age_stats"], "seed": config.seed,
            "weights_variant": payload.get("weights_variant", "raw"),
            "source_files_sha256_lf": source_hashes()}
    encoded = (json.dumps(spec, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    if len(encoded) + checkpoint.stat().st_size > DEPLOYMENT_LIMIT:
        raise ValueError("Checkpoint plus inference spec exceeds 500MB")
    write_json_exclusive(destination, spec)
    return spec


class InferenceSession:
    def __init__(self, spec_file, device="cuda"):
        self.spec_file = Path(spec_file).resolve()
        self.spec = json.loads(self.spec_file.read_text(encoding="utf-8"))
        spec = self.spec
        if (spec.get("format_version") != 1 or spec.get("kind") != "glaucoma_single_image"
                or spec.get("precision") not in ("fp16", "fp32")
                or spec.get("tta") not in ("none", "hflip") or spec.get("threshold") != 0.5):
            raise ValueError("Invalid frozen inference spec")
        if spec.get("source_files_sha256_lf") != source_hashes():
            raise ValueError("Inference source changed; review and freeze a new spec")
        self.checkpoint = (self.spec_file.parent / spec["checkpoint"]["file"]).resolve()
        if (self.checkpoint.stat().st_size != spec["checkpoint"]["bytes"]
                or sha256_file(self.checkpoint) != spec["checkpoint"]["sha256"]):
            raise ValueError("Inference checkpoint SHA256/size mismatch")
        self.deployment = [
            {"file": str(self.checkpoint), "bytes": self.checkpoint.stat().st_size,
             "sha256": spec["checkpoint"]["sha256"]},
            {"file": str(self.spec_file), "bytes": self.spec_file.stat().st_size,
             "sha256": sha256_file(self.spec_file)},
        ]
        self.deployment_bytes = sum(item["bytes"] for item in self.deployment)
        if self.deployment_bytes > DEPLOYMENT_LIMIT:
            raise ValueError("Actual loaded weights and required spec exceed 500MB")
        self.device = resolve_device(device)
        if spec["precision"] == "fp16" and self.device.type != "cuda":
            raise ValueError("fp16 spec requires CUDA; freeze a separate fp32 spec for CPU diagnostics")
        seed_everything(spec["seed"])
        self.model, self.config, self.age_stats = load_model(self.checkpoint, self.device)
        if sha256_file(self.checkpoint) != spec["checkpoint"]["sha256"]:
            raise ValueError("Inference checkpoint changed while loading")
        if (self.config.model.__dict__ != spec["model"] or preprocess_spec(self.config) != spec["preprocessing"]
                or self.age_stats.to_dict() != spec["age_stats"]):
            raise ValueError("Spec and checkpoint configuration differ")
        self.model.requires_grad_(False).eval()
        preprocessing = dict(spec["preprocessing"])
        self.transform = build_transform(preprocessing.pop("image_size"), **preprocessing)

    @property
    def needs_age(self):
        return (self.config.model.name == "convnext_tiny_age_residual"
                or self.config.model.params.get("use_age", False))

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @torch.inference_mode()
    def predict_image(self, sample):
        # File decode, transform, host/device transfer and postprocessing belong in the timer.
        image = self.transform(load_rgb(sample.path)).unsqueeze(0).to(self.device)
        age = torch.tensor([self.age_stats.transform(sample.age)], dtype=torch.float32, device=self.device)
        probabilities = []
        views = (image,) if self.spec["tta"] == "none" else (image, image.flip(-1))
        with torch.autocast(device_type=self.device.type, dtype=torch.float16,
                            enabled=self.spec["precision"] == "fp16"):
            for view in views:
                logit = self.model(view, age).float()
                if logit.shape != (1,) or not torch.isfinite(logit).all():
                    raise ValueError("Model must return one finite logit")
                probabilities.append(logit.sigmoid())
        probability = torch.stack(probabilities).mean().item()
        return probability, int(probability >= self.spec["threshold"])


def predict_and_write(session, sample, writer, stream):
    probability, label = session.predict_image(sample)
    serialized = format(probability, ".10g")
    writer.writerow([sample.name, serialized, label])
    stream.flush()
    return float(serialized), label


def predict_samples(spec_file, samples, output, device="cuda"):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if Path(str(output) + ".provenance.json").exists():
        raise FileExistsError(str(output) + ".provenance.json")
    session = InferenceSession(spec_file, device)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(SUBMISSION_FIELDS)
        handle.flush()
        for sample in samples:
            predict_and_write(session, sample, writer, handle)
            count += 1
    report = {"count": count, "precision": session.spec["precision"], "tta": session.spec["tta"],
              "batch_size": 1, "spec_sha256": sha256_file(spec_file),
              "prediction_sha256": sha256_file(output), "deployment_files": session.deployment,
              "deployment_bytes": session.deployment_bytes}
    write_json_exclusive(str(output) + ".provenance.json", report)
    return report


def read_submission(path):
    records = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != SUBMISSION_FIELDS:
            raise ValueError(f"Submission columns must be {SUBMISSION_FIELDS}")
        for row in reader:
            name = row["Image_name"]
            if not name or name in (".", "..") or Path(name).name != name or "/" in name or "\\" in name:
                raise ValueError("Invalid image name in submission")
            value = float(row["Probability"])
            if not math.isfinite(value) or not 0 <= value <= 1 or row["Pred_Label"] not in ("0", "1"):
                raise ValueError("Submission requires finite probabilities and binary labels")
            if name.casefold() in records:
                raise ValueError("Duplicate image name in submission")
            records[name.casefold()] = (name, value, int(row["Pred_Label"]))
    if not records:
        raise ValueError("Empty submission")
    return records


def evaluate_frozen(spec_file, samples, output_dir, device="cuda"):
    """Evaluate labelled train/valid rows through the exact serialized submission path."""
    from .metrics import age_stratified_metrics, binary_metrics
    samples = list(samples)
    if not samples or any(sample.label not in (0, 1) for sample in samples):
        raise ValueError("Frozen evaluation requires labelled train/valid rows")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    provenance = predict_samples(spec_file, samples, output_dir / "probabilities.csv", device)
    reference = read_submission(output_dir / "probabilities.csv")
    probabilities = [reference[sample.name.casefold()][1] for sample in samples]
    labels, ages = [sample.label for sample in samples], [sample.age for sample in samples]
    metrics = binary_metrics(labels, probabilities)
    metrics["age_groups"] = age_stratified_metrics(labels, probabilities, ages)
    metrics["evaluation_path"] = "frozen_batch1_serialized_probability"
    metrics["spec_sha256"] = provenance["spec_sha256"]
    with (output_dir / "predictions.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Image_name", "True_Label", "Probability"])
        for sample, probability in zip(samples, probabilities):
            writer.writerow([sample.name, sample.label, format(probability, ".10g")])
    write_json_exclusive(output_dir / "metrics.json", metrics)
    return metrics
