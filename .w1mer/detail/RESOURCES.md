# Shared-server resource rules (agreed 2026-10-22)

- **Our GPUs: 2 and 3 only.** GPUs 0, 1, 4, 5 are reserved for teammates'
  planned work; 6 and 7 belong to unrelated services. Never launch on another
  card even if it looks idle. Re-check `nvidia-smi` before every launch.
- GPU 2 or 3 may be busy with a teammate job; wait for it instead of sharing.
- Data caches live in `/dev/shm/ncccu-cache384` (24 GB tmpfs) to avoid the
  5400 rpm HDD that made four concurrent arms thrash (swap 100 %, 40+ D-state
  processes, ~18 img/s per arm). Disk copy: `/var/tmp/ncccu-heyuhang/`.
- Dataloader workers: 4 per run. Do not raise them on this host.
- The host has 62 GB RAM and an already-full 8 GB swap; keep total RSS low.
- `/home/heyuhang/work/ncccu` is our run mirror; `/var/tmp/ncccu-heyuhang` is
  ours; never write under `/home/lichunyu` or `/home/zhenxiaojun`.
