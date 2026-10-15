"""Synthetic NCCL correctness, latency and sustained-communication checks.

No dataset, optimizer, training loop or model weights. Launch with torchrun.
Reported latency includes host launch and GPU completion; it is not model throughput.
"""

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import statistics
import time

import torch
import torch.distributed as dist


def verify(values, expected):
    if not (values == expected).all().item():
        raise RuntimeError(f"Collective result differs from {expected}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes-mib", nargs="+", type=float,
                        default=[1 / 256, 1, 16, 64, 256])
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--seconds", type=float, default=120)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations < 1 or args.seconds < 0 or args.timeout < 1 or min(args.sizes_mib) <= 0:
        parser.error("Positive sizes, iterations, timeout and nonnegative seconds required")
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=args.timeout))
    results = []
    try:
        for dtype in (torch.float32, torch.float16):
            item_size = torch.empty((), dtype=dtype).element_size()
            for mib in args.sizes_mib:
                count = max(1, round(mib * 1024**2 / item_size))
                values = torch.empty(count, dtype=dtype, device=device)
                for operation in ("all_reduce", "broadcast"):
                    timings = []
                    dist.barrier()
                    for index in range(args.iterations + 5):
                        offset = index % 7
                        source = index % world
                        values.fill_(rank + 1 + offset)
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        if operation == "all_reduce":
                            dist.all_reduce(values)
                            expected = world * (world + 1) / 2 + world * offset
                        else:
                            dist.broadcast(values, src=source)
                            expected = source + 1 + offset
                        torch.cuda.synchronize()
                        elapsed_ms = (time.perf_counter() - start) * 1000
                        verify(values, expected)
                        if index >= 5:
                            timings.append(elapsed_ms)
                    slowest = torch.tensor(timings, dtype=torch.float64, device=device)
                    dist.all_reduce(slowest, op=dist.ReduceOp.MAX)
                    if rank == 0:
                        times = slowest.cpu().tolist()
                        row = dict(operation=operation, dtype=str(dtype), bytes=count * item_size,
                                   iterations=args.iterations, mean_ms=statistics.mean(times),
                                   median_ms=statistics.median(times), max_ms=max(times),
                                   payload_GB_per_s=count * item_size / (statistics.mean(times) * 1e6))
                        results.append(row)
                        print("[COMM-CHECK] " + json.dumps(row), flush=True)
                del values

        # Rank 0 decides the loop length and broadcasts it, so clocks cannot cause
        # ranks to enter a different number of collectives at the time boundary.
        values = torch.empty(16 * 1024**2 // 4, dtype=torch.float32, device=device)
        keep_going = torch.zeros(1, dtype=torch.int32, device=device)
        cycles = 0
        dist.barrier()
        start = time.monotonic()
        last_report = start
        while True:
            if rank == 0:
                keep_going.fill_(time.monotonic() - start < args.seconds)
            dist.broadcast(keep_going, src=0)
            if not keep_going.item():
                break
            for _ in range(10):
                offset = cycles % 7
                values.fill_(rank + 1 + offset)
                dist.all_reduce(values)
                verify(values, world * (world + 1) / 2 + world * offset)
                values.fill_(rank + 1 + offset)
                source = cycles % world
                dist.broadcast(values, src=source)
                verify(values, source + 1 + offset)
                cycles += 1
            if rank == 0 and time.monotonic() - last_report >= 20:
                print(f"[COMM-CHECK] sustained: {cycles * 2} checked payload collectives, "
                      f"{time.monotonic() - start:.1f}s", flush=True)
                last_report = time.monotonic()
        elapsed = time.monotonic() - start
        if rank == 0:
            report = dict(torch=torch.__version__, cuda=torch.version.cuda,
                          nccl=torch.cuda.nccl.version(), world_size=world,
                          visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                          p2p_disabled=os.environ.get("NCCL_P2P_DISABLE"),
                          results=results, stress_seconds=elapsed,
                          stress_payload_collectives=cycles * 2,
                          stress_message_bytes=values.numel() * values.element_size(),
                          timing_definition="Slowest rank per sample; host launch to GPU completion; "
                                            "input fill and result check excluded; 5 warmups excluded")
            with args.output.open("x", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2)
        dist.barrier()
        print(f"[COMM-CHECK] rank={rank} ALL CHECKS PASSED", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
