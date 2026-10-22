#!/usr/bin/env bash
# Launch one training run detached and record an exit-code sentinel.
# Usage: scripts/run_one.sh <config.json> <physical_gpu> [tag] [image_cache]
set -euo pipefail

CFG=${1:?config}
GPU=${2:?gpu index}
TAG=${3:-$(basename "$CFG" .json)}
CACHE=${4:-}

ROOT=/home/heyuhang/work/ncccu
PY=/home/lichunyu/miniconda3/envs/glaucoma/bin/python
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT=$ROOT/outputs/runs/${TAG}_${STAMP}
mkdir -p "$OUT"

EXTRA=()
if [ -n "$CACHE" ]; then
  EXTRA+=(--image-cache "$CACHE")
fi

cd "$ROOT"
nohup env \
  CUDA_VISIBLE_DEVICES="$GPU" \
  PYTHONPATH=src \
  PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=2 \
  bash -c "\"$PY\" -u -m glaucoma train --config \"$CFG\" --device cuda:0 \
    --output-root \"$OUT\" ${EXTRA[*]:-} > \"$OUT/train.log\" 2>&1; \
    echo \$? > \"$OUT/exit\"" \
  > /dev/null 2>&1 &
echo $! > "$OUT/train.pid"
printf '%s\n' "$OUT"
