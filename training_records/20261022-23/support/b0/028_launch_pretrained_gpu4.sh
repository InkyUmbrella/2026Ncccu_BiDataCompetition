#!/usr/bin/env bash
set -uo pipefail
cd /home/lichunyu/competition-pretrained
export CUDA_VISIBLE_DEVICES=4
export PYTHONPATH=/home/lichunyu/competition-pretrained/src
export OMP_NUM_THREADS=2
trap 'rc=$?; printf "%s\n" "$rc" > /home/lichunyu/competition-pretrained/outputs/pretrained-validation/migrations/gpu2-to-gpu4_20261022T142431Z/train_gpu4.exit' EXIT
/home/lichunyu/miniconda3/envs/glaucoma/bin/python -u /home/lichunyu/competition-pretrained/outputs/pretrained-validation/migrations/gpu2-to-gpu4_20261022T142431Z/resume_gpu4.py
