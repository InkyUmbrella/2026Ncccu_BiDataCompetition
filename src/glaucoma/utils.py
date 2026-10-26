from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import uuid

import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def resolve_device(name):
    device = torch.device(name)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Only cpu and cuda devices are supported")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu for diagnostics")
    if device.type == "cuda":
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)
    return device


def new_run_directory(root, experiment, seed):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = Path(root) / experiment / f"seed{seed}_{stamp}_{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def atomic_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            writer(handle)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_json(path, value):
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
    atomic_write(path, lambda handle: handle.write(payload))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def environment_metadata():
    metadata = {"python": sys.version, "python_executable": sys.executable,
                "platform": platform.platform(), "torch": str(torch.__version__),
                "cuda_runtime": torch.version.cuda, "cudnn_version": torch.backends.cudnn.version(),
                "packages": {}, "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "cudnn_benchmark": torch.backends.cudnn.benchmark}
    for name in ("torchvision", "numpy", "scikit-learn", "Pillow"):
        metadata["packages"][name] = importlib.metadata.version(name)
    metadata["installed_packages"] = {
        item.metadata["Name"]: item.version for item in importlib.metadata.distributions()
        if item.metadata["Name"]}
    try:
        metadata["git_revision"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL, timeout=5).strip()
        metadata["git_status"] = subprocess.check_output(
            ["git", "status", "--short"], text=True, stderr=subprocess.DEVNULL, timeout=5).strip()
    except (OSError, subprocess.SubprocessError):
        metadata["git_revision"] = None
    source = hashlib.sha256()
    root = Path(__file__).parent
    for path in sorted(root.rglob("*.py")):
        source.update(path.relative_to(root).as_posix().encode())
        source.update(path.read_bytes())
    metadata["source_sha256"] = source.hexdigest()
    return metadata
