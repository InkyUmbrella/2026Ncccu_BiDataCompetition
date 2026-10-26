#!/bin/bash
set -euo pipefail
RUN=/home/lichunyu/competition-fwq-pretrained/outputs/fwq-transfer/transfer_20261022T134645Z
cd /home/lichunyu/competition-fwq-pretrained
export CUDA_VISIBLE_DEVICES=5
export OMP_NUM_THREADS=2
export PYTHONPATH="$PWD/src"
trap 'rc=$?; printf "%s\n" "$rc" > "$RUN/transfer.exit"' EXIT
run_stage() {
  local phase="$1"; shift
  printf '%s\n' "$phase" > "$RUN/stage"
  if "$@" > "$RUN/$phase.log" 2>&1; then rc=0; else rc=$?; fi
  printf '%s\n' "$rc" > "$RUN/$phase.exit"
  if [ "$rc" -ne 0 ]; then exit "$rc"; fi
}
run_stage b0_evaluation /home/lichunyu/miniconda3/envs/glaucoma/bin/python -u -m glaucoma evaluate --checkpoint /home/lichunyu/competition-fwq-pretrained/outputs/fwq-transfer/transfer_20261022T134645Z/b0_snapshot.pt --data-root /home/lichunyu/competition/dataset/glaucoma_first --device cuda:0 --batch-size 32 --num-workers 4 --split valid --output-dir /home/lichunyu/competition-fwq-pretrained/outputs/fwq-transfer/transfer_20261022T134645Z/b0_evaluation
run_stage b0_verification /home/lichunyu/miniconda3/envs/glaucoma/bin/python /home/lichunyu/competition-fwq-pretrained/outputs/fwq-transfer/transfer_20261022T134645Z/verify_transfer.py b0
run_stage gpu_check_before_age /home/lichunyu/miniconda3/envs/glaucoma/bin/python /home/lichunyu/competition-fwq-pretrained/outputs/fwq-transfer/transfer_20261022T134645Z/guard_gpu.py 5
run_stage age_training /home/lichunyu/miniconda3/envs/glaucoma/bin/python -u -m glaucoma train --config configs/experiments/b0_crop384_age_residual_imagenet.json --data-root /home/lichunyu/competition/dataset/glaucoma_first --device cuda:0 --image-cache /var/tmp/glaucoma-pretrained-lichunyu-crop384-20261022 --init-from /home/lichunyu/competition-fwq-pretrained/outputs/fwq-transfer/transfer_20261022T134645Z/b0_snapshot.pt --output-root /home/lichunyu/competition-fwq-pretrained/outputs/fwq-transfer/transfer_20261022T134645Z/age_training
run_stage age_verification /home/lichunyu/miniconda3/envs/glaucoma/bin/python /home/lichunyu/competition-fwq-pretrained/outputs/fwq-transfer/transfer_20261022T134645Z/verify_transfer.py age
printf "complete\n" > "$RUN/stage"
