#!/usr/bin/env bash
# Push the current branch to the shared bare repo and run the CPU test suite
# on the GPU server. Usage: scripts/dev_test.sh [extra unittest args]
set -euo pipefail
cd "$(dirname "$0")/.."
KEY=${SSH_KEY:-/home/niko/.ssh/id_ed25519_xianka}
HOST=${SSH_HOST:-heyuhang@100.68.97.6}
BRANCH=$(git rev-parse --abbrev-ref HEAD)
GIT_SSH_COMMAND="ssh -i $KEY -o StrictHostKeyChecking=no" \
  git push -q server "HEAD:${BRANCH}"
ssh -i "$KEY" "$HOST" "cd /home/heyuhang/work/ncccu && \
  git fetch -q origin && git checkout -q ${BRANCH} && git pull -q origin ${BRANCH} && \
  PYTHONPATH=src OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 \
  /home/lichunyu/miniconda3/envs/glaucoma/bin/python -B -m unittest discover -s tests -v 2>&1 | tail -6"
