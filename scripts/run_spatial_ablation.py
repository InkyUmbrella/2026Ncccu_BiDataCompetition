"""Matched residual architecture experiments from the verified round2 seed42.

Reuses existing lossless caches. No new pretrained downloads; no other GPUs.
The round2 plain-refine arm is the unchanged training recipe control.
"""
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
import os
from pathlib import Path
import subprocess
import traceback

import run_round2_server as runner

WORK_CREATED = False

def main(args):
    global WORK_CREATED
    os.umask(0o077)
    runner.WORK = Path(args.work_root).resolve()
    runner.WORK.mkdir(parents=True, exist_ok=False)
    WORK_CREATED = True
    previous_root = Path(args.previous_work).resolve()
    previous = json.loads((previous_root / 'state.json').read_text())
    reference = previous['jobs']['seed42']
    if reference['status'] != 'complete':
        raise RuntimeError('Need independently verified seed42 reference')
    checkpoint = Path(reference['result']['artifact']['path'])
    if runner.sha(checkpoint) != reference['result']['artifact']['sha256']:
        raise RuntimeError('Reference weight hash mismatch')
    runner.DATA = previous['data_root']
    cache = previous['caches']['512']
    if not (Path(cache) / 'manifest.json').exists():
        raise RuntimeError('Existing lossless cache is unavailable')
    runner.state(phase='running', started_at=runner.stamp(),
                 source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=runner.ROOT,text=True).strip(),
                 csv_hashes=previous['csv_hashes'], data_root=runner.DATA,
                 reference=reference['result'], control=previous['jobs'].get('seed42_refine'),
                 gpus=[3,4], cache=cache)
    if previous['jobs'].get('seed43',{}).get('status')=='complete':
        (runner.WORK/'archive').mkdir(exist_ok=True)
        runner.run([runner.PYTHON,str(runner.ROOT/'scripts/check_complementarity.py'),
                    str(previous_root/'jobs/seed42/evaluation/predictions.csv'),
                    str(previous_root/'jobs/seed43/evaluation/predictions.csv'),
                    '--output',str(runner.WORK/'archive/convnext_complementarity.json')],
                   runner.WORK/'complementarity.log')

    def arm(spec):
        name, gpu, config = spec
        try:
            runner.experiment(name, config, gpu, cache, checkpoint)
        except Exception:
            runner.job_state(name, status='failed', error=traceback.format_exc())

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(arm, [('spatial',3,'shimmer_512_spatial_refine_seed42.json'),
                            ('age',4,'shimmer_512_age_refine_seed42.json')]))
    results = [v['result'] for v in runner.STATE['jobs'].values() if v.get('status')=='complete']
    summary = {'results':results,'reference':reference['result'],
               'control':json.loads((previous_root/'state.json').read_text())['jobs'].get('seed42_refine'),
               'failed_jobs':{k:v['error'] for k,v in runner.STATE['jobs'].items() if v.get('status')=='failed'}}
    runner.save(runner.WORK/'archive/summary.json',summary)
    runner.state(phase='finished', finished=True, summary=summary)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-root',required=True)
    parser.add_argument('--previous-work',required=True)
    args=parser.parse_args()
    try:
        main(args)
    except Exception:
        # Only write in a work directory created by this invocation.
        if WORK_CREATED:
            runner.save(runner.WORK/'archive/error.json',{'error':traceback.format_exc()})
            runner.state(phase='failed',finished=True,error=traceback.format_exc())
        raise
