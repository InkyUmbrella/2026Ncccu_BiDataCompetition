#!/bin/bash
cd /home/lichunyu/competition-fwq-pretrained || exit 1
export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=2
export PYTHONPATH="$PWD/src"
/home/lichunyu/miniconda3/envs/glaucoma/bin/python -u -c 'import torch,unittest; torch.set_num_threads(2); result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover("tests")); raise SystemExit(not result.wasSuccessful())'
status=$?
printf '%s\n' "$status" > outputs/fwq-transfer/integration-tests.exit
exit "$status"
