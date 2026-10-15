"""Isolate CUDA, NCCL collectives and DDP startup without reading any images.

Run on the server with torchrun, one process per allocated GPU. Run an external
timeout as well: not every driver or communicator initialization stall is
guaranteed to be interrupted by a process-group timeout.
"""

import argparse
import os
from datetime import datetime, timedelta

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel


def report(rank, message):
    print(f"[DEBUG-ddp-probe] {datetime.now():%H:%M:%S} rank={rank} {message}", flush=True)


def probe(device, backend, rank, world_size, timeout):
    report(rank, f"torch={torch.__version__}; backend={backend}; device={device}; "
                 f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '(unset)')}; "
                 f"NCCL_P2P_DISABLE={os.environ.get('NCCL_P2P_DISABLE', '(unset)')}")
    report(rank, "local tensor calculation: START")
    x = torch.ones((32, 32), device=device)
    if (x @ x).sum().item() != 32768:
        raise RuntimeError("Local tensor calculation returned an incorrect result")
    report(rank, "local tensor calculation: PASS")

    report(rank, "init_process_group: START")
    dist.init_process_group(backend, timeout=timedelta(seconds=timeout))
    report(rank, "init_process_group: RETURNED (first collective may initialize transport)")

    report(rank, "all_reduce: START")
    values = torch.full((4096,), float(rank + 1), device=device)
    dist.all_reduce(values)
    expected_sum = world_size * (world_size + 1) / 2
    torch.testing.assert_close(values, torch.full_like(values, expected_sum))
    report(rank, f"all_reduce: PASS (sum={expected_sum:g})")

    report(rank, "DDP constructor: START")
    model = nn.Linear(4, 1, bias=False).to(device)
    model = DistributedDataParallel(
        model, device_ids=[device.index] if device.type == "cuda" else None
    )
    report(rank, "DDP constructor: PASS")
    report(rank, "DDP forward/backward: START")
    inputs = torch.full((2, 4), float(rank + 1), device=device)
    model(inputs).mean().backward()
    gradient = model.module.weight.grad
    torch.testing.assert_close(gradient, torch.full_like(gradient, (world_size + 1) / 2))
    report(rank, "DDP forward/backward: PASS")
    dist.barrier()
    report(rank, "ALL CHECKS PASSED")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    if "RANK" not in os.environ:
        parser.error("Launch this script with torchrun")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    try:
        probe(device, "nccl" if device.type == "cuda" else "gloo",
              rank, world_size, args.timeout)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
