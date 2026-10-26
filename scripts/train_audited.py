"""Portable foreground launcher: retain the full console log, command and exit code."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", default="outputs/audited")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-cache")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--stop-after-epochs", type=int)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--resume")
    source.add_argument("--init-from")
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    config = Path(args.config).resolve()
    output = Path(args.output_root).resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = output / "launches" / f"{stamp}_{uuid.uuid4().hex[:8]}"
    directory.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, "-u", "-m", "glaucoma", "train", "--config", str(config),
               "--data-root", str(Path(args.data_root).resolve()), "--output-root", str(output),
               "--device", args.device]
    for name in ("image_cache", "resume", "init_from"):
        value = getattr(args, name)
        if value:
            command += ["--" + name.replace("_", "-"), str(Path(value).resolve())]
    for name in ("seed", "stop_after_epochs"):
        value = getattr(args, name)
        if value is not None:
            command += ["--" + name.replace("_", "-"), str(value)]
    record = {"started_at_utc": datetime.now(timezone.utc).isoformat(), "command": command,
              "cwd": str(repository), "config_file_sha256": digest(config),
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "log": str(directory / "train.log"), "run_directory": None, "exit_code": None}
    save(directory / "launch.json", record)
    environment = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    environment["PYTHONPATH"] = str(repository / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    print(f"[launch] {directory}", flush=True)
    with (directory / "train.log").open("wb") as log:
        child = subprocess.Popen(command, cwd=repository, env=environment,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        record["pid"] = child.pid
        save(directory / "launch.json", record)
        try:
            for line in child.stdout:
                log.write(line)
                log.flush()
                sys.stdout.buffer.write(line)
                sys.stdout.buffer.flush()
                message = line.decode("utf-8", errors="replace").strip()
                if message.startswith("[run] "):
                    record["run_directory"] = message[6:]
                    save(directory / "launch.json", record)
            code = child.wait()
        except KeyboardInterrupt:
            child.terminate()
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            code = 130
    record.update(exit_code=code, finished_at_utc=datetime.now(timezone.utc).isoformat(),
                  train_log_sha256=digest(directory / "train.log"))
    (directory / "train.exit").write_text(str(code) + "\n", encoding="utf-8")
    save(directory / "launch.json", record)
    print(f"[launch] exit={code}; evidence={directory}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
