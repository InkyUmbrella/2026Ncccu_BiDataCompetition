"""100 images, batch one, three continuous rounds, with five separate warmups."""

import csv
import importlib.metadata
import math
import platform
from pathlib import Path
import subprocess
import time

import torch

from .data.age import parse_age
from .data.dataset import Sample
from .inference import (InferenceSession, SUBMISSION_FIELDS, predict_and_write,
                        read_submission, write_json_exclusive)
from .utils import sha256_file


def picture_files(directory, expected_count):
    directory = Path(directory).resolve()
    files = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in (".jpg", ".jpeg", ".png"):
            if not path.resolve().is_relative_to(directory):
                raise ValueError("Images must be inside the supplied directory")
            key = path.name.casefold()
            if key in files:
                raise ValueError(f"Ambiguous image name in {directory}: {path.name}")
            files[key] = path
    if len(files) != expected_count:
        raise ValueError(f"Expected {expected_count} images in {directory}, got {len(files)}")
    return files


def read_ages(path):
    if path is None:
        return {}
    result = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"Image_name", "Age"}.issubset(reader.fieldnames or []):
            raise ValueError("Age CSV requires Image_name and Age")
        for row in reader:
            key = row["Image_name"].casefold()
            if key in result:
                raise ValueError("Duplicate image in age CSV")
            result[key] = parse_age(row["Age"])
    return result


def hardware_environment(session):
    driver = None
    if session.device.type == "cuda":
        try:
            result = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                    capture_output=True, text=True, timeout=5, check=True)
            driver = result.stdout.splitlines()[0].strip()
        except (OSError, subprocess.SubprocessError, IndexError):
            pass
    cudnn = torch.backends.cudnn.version()
    version = lambda value: tuple(int(part) for part in value.split("."))
    actual = {"gpu": torch.cuda.get_device_name(session.device) if session.device.type == "cuda" else None,
              "driver": driver, "python": platform.python_version(), "pytorch": str(torch.__version__),
              "torchvision": importlib.metadata.version("torchvision"), "cuda": torch.version.cuda,
              "cudnn": cudnn}
    compatible = (actual["gpu"] is not None and "T4" in actual["gpu"]
                  and actual["python"].startswith("3.12.")
                  and actual["pytorch"].split("+")[0] == "2.10.0"
                  and actual["torchvision"].split("+")[0] == "0.25.0"
                  and actual["cuda"] == "12.8" and cudnn is not None and cudnn // 10000 == 9
                  and driver is not None and version(driver) >= version("570.117"))
    return actual, bool(compatible)


def run_cost(spec_file, submission, pictures_100, pictures_5, output_dir,
             device="cuda", ages_csv=None, diagnostic=False):
    directory = Path(output_dir)
    if directory.exists():
        raise FileExistsError(directory)
    measured, warmup = picture_files(pictures_100, 100), picture_files(pictures_5, 5)
    if set(measured) & set(warmup):
        raise ValueError("Warmup and measured images must be separate")
    reference = read_submission(submission)
    if not set(measured).issubset(reference):
        raise ValueError("All 100 measured images must occur in submission.csv")
    ages = read_ages(ages_csv)
    session = InferenceSession(spec_file, device)
    if session.needs_age and not (set(measured) | set(warmup)).issubset(ages):
        raise ValueError("This age model requires an age CSV covering measured and warmup images")
    environment, compatible = hardware_environment(session)
    if not diagnostic and not compatible:
        raise ValueError("Specified T4/software environment does not match; use --diagnostic for engineering checks")
    # Keep the same order and spelling as the actual submitted CSV.
    samples = [Sample(name, measured[key], ages.get(key, float("nan")), None)
               for key, (name, _, _) in reference.items() if key in measured]
    warms = [Sample(path.name, path, ages.get(key, float("nan")), None) for key, path in warmup.items()]
    directory.mkdir(parents=True, exist_ok=False)
    with (directory / "warmup.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(SUBMISSION_FIELDS)
        for sample in warms:
            predict_and_write(session, sample, writer, handle)
        session.synchronize()
    durations, mismatches = {sample.name: [] for sample in samples}, {}
    for round_index in range(3):
        with (directory / f"round_{round_index + 1}.csv").open("x", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(SUBMISSION_FIELDS)
            handle.flush()
            for sample in samples:
                session.synchronize()
                started = time.perf_counter()
                probability, label = predict_and_write(session, sample, writer, handle)
                session.synchronize()
                elapsed = (time.perf_counter() - started) * 1000
                durations[sample.name].append(elapsed)
                expected = reference[sample.name.casefold()]
                if probability != expected[1] or label != expected[2]:
                    mismatches.setdefault(sample.name, []).append(round_index + 1)
                print(f"[cost] round={round_index + 1} image={sample.name} ms={elapsed:.3f}", flush=True)
    means = {name: sum(values) / 3 for name, values in durations.items()}
    average = sum(means.values()) / 100
    if not math.isfinite(average):
        raise ValueError("Non-finite timing measurement")
    report = {"count": 100, "warmup_count": 5, "rounds": 3, "batch_size": 1,
              "average_ms": average, "per_image_round_ms": durations, "per_image_mean_ms": means,
              "mismatch_count": len(mismatches), "mismatch_rounds": mismatches,
              "comparison": "Exact numeric equality after shared .10g CSV serialization, plus label equality",
              "precision": session.spec["precision"], "tta": session.spec["tta"],
              "deployment_files": session.deployment, "deployment_bytes": session.deployment_bytes,
              "spec_sha256": sha256_file(spec_file), "submission_sha256": sha256_file(submission),
              "actual_environment": environment, "target_gpu_software_matches": compatible,
              "cpu_ram_ssd_limits_verified": False, "diagnostic": diagnostic,
              "within_latency_limit": average <= 100, "predictions_match": not mismatches,
              "organizer_retest_required": True}
    write_json_exclusive(directory / "cost_report.json", report)
    print(f"[CONCLUSION] 100 images, batch=1, 3 rounds; average={average:.3f} ms; "
          f"mismatches={len(mismatches)}; precision={session.spec['precision']}; "
          f"diagnostic={diagnostic}; T4/software_match={compatible}", flush=True)
    return report
