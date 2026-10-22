#!/usr/bin/env bash
# Summarise our runs: epoch progress, best valid AUROC, exit status.
set -uo pipefail
ROOT=${1:-/home/heyuhang/work/ncccu/outputs/runs}
for d in "$ROOT"/*/; do
  [ -f "$d/train.log" ] || continue
  last=$(grep -E '^\[epoch ' "$d/train.log" | tail -1)
  best=$(grep -oE 'valid_auc=[0-9.]+' "$d/train.log" | sed 's/valid_auc=//' | sort -g | tail -1)
  exit=$(cat "$d/exit" 2>/dev/null || echo running)
  printf '%-55s exit=%-7s best=%-8s %s\n' "$(basename "$d")" "$exit" "${best:-?}" "${last:-starting}"
done
