"""Switch this user's jobs at epoch checkpoints, retaining rollback until verified.

Linux only. Run from the training checkout with its Python environment.
"""

import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


def identity(pid):
    try:
        path = Path(f"/proc/{pid}")
        if path.stat().st_uid != os.getuid():
            return None
        # Field 22 is starttime; split after comm, which may contain spaces.
        stat = (path / "stat").read_text().rsplit(")", 1)[1].split()
        return None if stat[0] == "Z" else stat[19]
    except FileNotFoundError:
        return None


def descendants(pid):
    result = {}
    for child in Path("/proc").iterdir():
        if not child.name.isdigit():
            continue
        try:
            stat = (child / "stat").read_text().rsplit(")", 1)[1].split()
            if int(stat[1]) == pid and identity(int(child.name)) is not None:
                child_pid = int(child.name)
                result[child_pid] = identity(child_pid)
                result.update(descendants(child_pid))
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return result


def send_verified(processes, sig):
    for pid, started in processes.items():
        if identity(pid) == started:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass


def terminate(processes):
    send_verified(processes, signal.SIGTERM)
    send_verified(processes, signal.SIGCONT)
    time.sleep(2)
    send_verified(processes, signal.SIGKILL)


def checkpoint_signature(path):
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


def available_gpus():
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used",
                                      "--format=csv,noheader,nounits"], text=True)
    return [int(line.split(",")[0]) for line in output.splitlines()
            if int(line.split(",")[0]) in (2, 3, 4, 5) and int(line.split(",")[1]) == 0]


def main():
    def cancelled(signum, frame):
        raise KeyboardInterrupt("Migration cancelled; restoring paused originals")

    signal.signal(signal.SIGTERM, cancelled)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--job", nargs=3, action="append", required=True,
                        metavar=("PID", "CONFIG", "LOG"))
    parser.add_argument("--state", required=True)
    args = parser.parse_args()
    root = Path.cwd().resolve()
    jobs = []
    for pid_text, config, log in args.job:
        pid = int(pid_text)
        started = identity(pid)
        command = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        if started is None or b"glaucoma" not in command or b"train" not in command:
            raise RuntimeError(f"Not an owned glaucoma training process: {pid}")
        if os.readlink(f"/proc/{pid}/cwd") != str(root) or config.encode() not in command:
            raise RuntimeError(f"Process checkout/config mismatch: {pid}")
        match = re.search(r"^\[run\] (.+)$", Path(log).read_text(), re.MULTILINE)
        if not match:
            raise RuntimeError(f"Cannot find run directory in {log}")
        checkpoint = Path(match.group(1)) / "last.pt"
        jobs.append(dict(old_pid=pid, old_started=started, config=config, log=log,
                         checkpoint=str(checkpoint), phase="waiting_cache"))

    def save():
        target = Path(args.state)
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(jobs, indent=2))
        temporary.replace(target)

    save()
    deadline = time.monotonic() + 7200
    while not (Path(args.cache) / "manifest.json").exists():
        if time.monotonic() > deadline:
            raise RuntimeError("Cache timeout; original jobs left running")
        time.sleep(5)
    benchmark = json.loads((root / "outputs/speed_benchmark.json").read_text())
    if benchmark["speedup"] < 1.1:
        raise RuntimeError("Cache benchmark did not improve throughput; originals left running")
    gpus = available_gpus()
    if len(gpus) < len(jobs):
        raise RuntimeError("Need idle GPUs 2-5 for rollback-safe verification; originals left running")
    from glaucoma.engine.checkpoint import read_checkpoint
    from glaucoma.config import Config
    from glaucoma.engine.trainer import _resume_compatible
    for job, gpu in zip(jobs, gpus):
        saved = read_checkpoint(job["checkpoint"], "training")
        config = Config.load(job["config"])
        config.device = "cuda:0"  # Match the CLI override used by both jobs.
        _resume_compatible(config, saved["config"])
        job.update(gpu=gpu, phase="waiting_epoch", baseline=list(checkpoint_signature(Path(job["checkpoint"]))))
    save()
    print("[switch] cache ready; waiting for next completed epoch", flush=True)
    try:
        while any(job["phase"] not in ("verified", "rolled_back") for job in jobs):
            for job in jobs:
                phase = job["phase"]
                if phase == "waiting_epoch":
                    if identity(job["old_pid"]) != job["old_started"]:
                        raise RuntimeError("Original training exited before switch")
                    if list(checkpoint_signature(Path(job["checkpoint"]))) == job["baseline"]:
                        continue
                    reserved = {other["gpu"] for other in jobs if other is not job}
                    idle = [gpu for gpu in available_gpus() if gpu not in reserved]
                    if not idle:
                        job["baseline"] = list(checkpoint_signature(Path(job["checkpoint"])))
                        print("[switch] no idle verification GPU; originals left running", flush=True)
                        save()
                        continue
                    if job["gpu"] not in idle:
                        job["gpu"] = idle[0]
                    parent = {job["old_pid"]: job["old_started"]}
                    send_verified(parent, signal.SIGSTOP)
                    job["paused"] = {**parent, **descendants(job["old_pid"])}
                    send_verified(job["paused"], signal.SIGSTOP)
                    job["phase"] = "starting"
                    save()
                    saved = read_checkpoint(job["checkpoint"], "training")
                    if saved["epoch"] + 1 >= saved["config"]["train"]["epochs"]:
                        send_verified(job["paused"], signal.SIGCONT)
                        job["phase"] = "rolled_back"
                        save()
                        continue
                    job["resume_epoch"] = saved["epoch"] + 1
                    env = os.environ.copy()
                    env.update(CUDA_VISIBLE_DEVICES=str(job["gpu"]), OMP_NUM_THREADS="2")
                    fast_log = str(Path(job["log"]).with_suffix(".cached.log"))
                    with open(fast_log, "xb", buffering=0) as log:
                        process = subprocess.Popen([sys.executable, "-u", "-m", "glaucoma", "train",
                            "--config", job["config"], "--device", "cuda:0", "--resume", job["checkpoint"],
                            "--image-cache", args.cache], env=env, stdin=subprocess.DEVNULL,
                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    job.update(new_pid=process.pid, new_started=identity(process.pid), fast_log=fast_log,
                               phase="verifying", launched=time.time())
                    print(f"[switch] PID {job['old_pid']} paused after epoch {job['resume_epoch']}; "
                          f"resumed PID {process.pid} on GPU {job['gpu']}", flush=True)
                    save()
                elif phase == "verifying":
                    text = Path(job["fast_log"]).read_text(errors="replace")
                    if (("[train] batch=100/" in text or "[epoch " in text)
                            and identity(job["new_pid"]) == job["new_started"] and "Traceback" not in text):
                        terminate(job["paused"])
                        job["phase"] = "verified"
                        Path(job["log"]).with_suffix(".pid").write_text(str(job["new_pid"]) + "\n")
                        print(f"[switch] PID {job['new_pid']} verified; old process released", flush=True)
                        save()
                    elif identity(job["new_pid"]) != job["new_started"] or "Traceback" in text or time.time() - job["launched"] > 1200:
                        terminate({job["new_pid"]: job["new_started"], **descendants(job["new_pid"])})
                        send_verified(job["paused"], signal.SIGCONT)
                        job["phase"] = "rolled_back"
                        print(f"[switch] startup failed; original PID {job['old_pid']} resumed", flush=True)
                        save()
            time.sleep(2)
    finally:
        for job in jobs:
            if job["phase"] not in ("verified", "rolled_back") and "paused" in job:
                if "new_pid" in job:
                    terminate({job["new_pid"]: job["new_started"], **descendants(job["new_pid"])})
                send_verified(job["paused"], signal.SIGCONT)


if __name__ == "__main__":
    main()
