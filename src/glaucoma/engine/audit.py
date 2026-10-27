"""Preserve the actual inputs and every epoch of a training invocation."""

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tarfile
from urllib.parse import urlparse

import torch
from torchvision.models import ConvNeXt_Tiny_Weights

from ..utils import atomic_write, save_json, sha256_file
from .checkpoint import inference_payload, save_checkpoint


def snapshot_sources(run_dir):
    package = Path(__file__).resolve().parents[1]
    repository = package.parents[1]
    paths = {f"src/glaucoma/{p.relative_to(package).as_posix()}": p
             for p in package.rglob("*.py")}
    for name in ("pyproject.toml", "requirements-server-tested.txt", "scripts/train_audited.py",
                 "predict.py", "cost.py", "scripts/freeze_inference.py", "scripts/prepare_dinov2.py"):
        path = repository / name
        if path.is_file():
            paths[name] = path
    for path in (repository / "licenses").rglob("*"):
        if path.is_file():
            paths[path.relative_to(repository).as_posix()] = path
    files = {name: sha256_file(path) for name, path in sorted(paths.items())}

    def archive(handle):
        with tarfile.open(fileobj=handle, mode="w:gz") as stream:
            for name, path in sorted(paths.items()):
                stream.add(path, arcname=name, recursive=False)

    destination = run_dir / "source.tar.gz"
    atomic_write(destination, archive)
    save_json(run_dir / "source_manifest.json", {
        "format_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": files, "archive_sha256": sha256_file(destination)})


def pretrained_source(config):
    if config.model.weights is None:
        return None
    if config.model.name == "dinov2_vits14":
        from ..models.pretrained import dinov2_source
        return dinov2_source()[0]
    weights = ConvNeXt_Tiny_Weights[config.model.weights]
    path = Path(torch.hub.get_dir()) / "checkpoints" / Path(urlparse(weights.url).path).name
    if not path.is_file():
        raise FileNotFoundError(f"Pretrained source must be retained for audit: {path}")
    return path


def record_initialization(run_dir, model, config, age_stats, start_epoch,
                          source_file=None, mode="new", parent_run=None):
    directory = run_dir / "initialization"
    directory.mkdir()
    initial = directory / "model_initial.pt"
    save_checkpoint(initial, inference_payload(model, config, age_stats, start_epoch - 1, {}))
    record = {"format_version": 1, "mode": mode, "seed": config.seed,
              "weights_declaration": config.model.weights, "parent_run_directory": parent_run,
              "model_initial": initial.relative_to(run_dir).as_posix(),
              "model_initial_sha256": sha256_file(initial), "source": None}
    source = Path(source_file).resolve() if source_file else pretrained_source(config)
    if source is not None:
        digest = sha256_file(source)
        copied = directory / ("source" + source.suffix)
        atomic_write(copied, lambda handle: _copy_into(source, handle))
        if sha256_file(copied) != digest:
            raise ValueError("Initialization source changed while being archived")
        record["source"] = {"original_path": str(source),
                            "file": copied.relative_to(run_dir).as_posix(), "sha256": digest}
    if mode == "new" and config.model.name == "dinov2_vits14" and config.model.weights:
        from ..models.pretrained import DINOV2_DOCUMENTS, dinov2_source
        _, manifest, documents = dinov2_source()
        for path in [manifest, *[documents / name for name in sorted(DINOV2_DOCUMENTS)]]:
            target = directory / path.name
            atomic_write(target, lambda handle, p=path: _copy_into(p, handle))
        record["pretrained_source_manifest"] = json.loads(manifest.read_text(encoding="utf-8"))
    save_json(run_dir / "initialization.json", record)


def _copy_into(path, handle):
    with path.open("rb") as source:
        shutil.copyfileobj(source, handle, length=8 * 1024 * 1024)


def save_epoch_checkpoint(run_dir, payload):
    epoch = payload["epoch"] + 1
    path = run_dir / "checkpoints" / f"epoch_{epoch:04d}.pt"
    if path.exists():
        raise FileExistsError(f"An epoch checkpoint must not be overwritten: {path}")
    save_checkpoint(path, payload)
    stat = path.stat()
    entry = {"epoch": epoch, "global_step": payload["global_step"],
             "file": path.relative_to(run_dir).as_posix(), "sha256": sha256_file(path),
             "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
             "saved_at_utc": datetime.now(timezone.utc).isoformat(),
             "best_epoch": payload["best"]["epoch"] + 1,
             "valid": payload["history"][-1]["valid"]}
    manifest = run_dir / "checkpoint_manifest.json"
    records = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {
        "format_version": 1, "config_sha256": sha256_file(run_dir / "config.json"),
        "initialization_sha256": sha256_file(run_dir / "initialization.json"),
        "source_manifest_sha256": sha256_file(run_dir / "source_manifest.json"), "epochs": []}
    records["epochs"].append(entry)
    save_json(manifest, records)
    # Keep last.pt as a regular, byte-identical recovery file on Linux and Windows.
    atomic_write(run_dir / "last.pt", lambda handle: _copy_into(path, handle))


def finalize_run_audit(run_dir):
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    epochs = json.loads((run_dir / "checkpoint_manifest.json").read_text(encoding="utf-8"))["epochs"]
    names = ("config.json", "age_stats.json", "metadata.json", "history.json", "metrics.json",
             "valid_predictions.csv", "best.pt", "last.pt", "source.tar.gz",
             "source_manifest.json", "initialization.json", "checkpoint_manifest.json")
    files = {name: sha256_file(run_dir / name) for name in names}
    if (run_dir / "data_audit.json").is_file():
        files["data_audit.json"] = sha256_file(run_dir / "data_audit.json")
    files.update({p.relative_to(run_dir).as_posix(): sha256_file(p)
                  for p in sorted((run_dir / "initialization").iterdir())})
    files.update({row["file"]: row["sha256"] for row in epochs})
    save_json(run_dir / "artifact_manifest.json", {
        "format_version": 1, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_complete": metrics["training_complete"],
        "completed_epochs": metrics["completed_epochs"],
        "recorded_epochs": [row["epoch"] for row in epochs], "files": files})


def verify_run_audit(run_dir):
    run_dir = Path(run_dir).resolve()
    manifest = json.loads((run_dir / "artifact_manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        path = (run_dir / name).resolve()
        if not path.is_relative_to(run_dir) or not path.is_file():
            raise ValueError(f"Missing or invalid artifact: {name}")
        if sha256_file(path) != expected:
            raise ValueError(f"Artifact SHA256 mismatch: {name}")
    return {"run_directory": str(run_dir), "verified_files": len(manifest["files"]),
            "training_complete": manifest["training_complete"],
            "completed_epochs": manifest["completed_epochs"],
            "recorded_epochs": manifest["recorded_epochs"]}
