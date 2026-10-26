#!/usr/bin/env bash
set -euo pipefail
cd /home/lichunyu/competition-pretrained
export PYTHONPATH=src OMP_NUM_THREADS=2
python_bin=/home/lichunyu/miniconda3/envs/glaucoma/bin/python
config=configs/experiments/b0_crop384_imagenet.json
data_root=/home/lichunyu/competition/dataset/glaucoma_first
cache_root=/var/tmp/glaucoma-pretrained-lichunyu-crop384-20261022
trap 'rc=$?; printf "[exit] code=%s\n" "$rc"; printf "%s\n" "$rc" > outputs/pretrained-validation/train.exit' EXIT
test -f "$cache_root/manifest.json"
used_mb=$(nvidia-smi --id=2 --query-gpu=memory.used --format=csv,noheader,nounits)
if (( used_mb > 128 )); then
  printf '[blocked] GPU 2 is occupied (%s MiB); training was not started.\n' "$used_mb"
  exit 3
fi
printf '[stage] reuse completed cache: %s\n' "$cache_root"
printf '[stage] fine-tune ImageNet ConvNeXt-Tiny on GPU 2, 30 epochs (device resolver fixed)\n'
CUDA_VISIBLE_DEVICES=2 "$python_bin" -u -m glaucoma train --config "$config" \
  --data-root "$data_root" --image-cache "$cache_root"
