"""Bounded, provenance-checked B0 improvement experiments on GPUs 4 and 5."""
import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
KEYS = ('auroc', 'ap', 'auprc_trapz')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    temporary = Path(str(path) + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_context(context):
    assert sha(context['source_snapshot']) == context['source_sha256']
    assert sha(Path(context['cache']) / 'manifest.json') == context['cache_manifest_sha256']
    assert sha(context['source_audit']) == context['audit_sha256']
    for split, digest in context['csv_hashes'].items():
        assert sha(Path(context['data_root']) / (split + '.csv')) == digest
    audit = read(context['source_audit'])
    assert not audit['errors'] and audit['image_source'] == 'lossless_cache'
    assert {key: value['count'] for key, value in audit['splits'].items()} == {'train': 51396, 'valid': 5849}
    return audit


def worker(root, name, resume=None):
    from glaucoma.config import Config
    from glaucoma.engine import trainer
    context = read(root / 'context.json')
    audit = verify_context(context)
    job = context['jobs'][name]
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == str(job['gpu'])
    config = Config.load(REPO / job['config'])
    config.data.root = context['data_root']
    config.output_root = str(root / 'runs' / name)
    config.device = 'cuda:0'

    def reuse(samples, *args, **kwargs):
        assert {key: len(value) for key, value in samples.items()} == {'train': 51396, 'valid': 5849}
        report = copy.deepcopy(audit)
        report['audit_reuse'] = {'source_report': context['source_audit'],
            'source_report_sha256': context['audit_sha256'],
            'cache_manifest_sha256': context['cache_manifest_sha256'],
            'verified_at_utc': datetime.now(timezone.utc).isoformat(),
            'reason': 'Same audited dataset/cache and unchanged preprocessing; normal cache loads still validate entries'}
        print('[audit] Reusing the verified full audit; all manifest and CSV hashes match', flush=True)
        return report

    trainer.audit_samples = reuse
    run = trainer.train(config, resume=resume, image_cache=context['cache'],
        init_from=context['source_snapshot'] if job['initialization'] == 'b0_best' and not resume else None)
    print('Results: ' + str(run.resolve()), flush=True)


def gpu_available(index):
    assert index in (4, 5)
    command = ['nvidia-smi', '--id=' + str(index), '--query-gpu=uuid,memory.used', '--format=csv,noheader,nounits']
    result = subprocess.run(command, capture_output=True, text=True, timeout=15, check=True)
    uuid, memory = result.stdout.strip().split(',')
    processes = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'],
        capture_output=True, text=True, timeout=15, check=True).stdout.splitlines()
    return int(memory.strip()) <= 256 and not any(line.split(',')[0].strip() == uuid.strip() for line in processes)


def verify_result(run, context, job):
    import numpy as np
    import torch
    from sklearn.metrics import roc_auc_score
    from glaucoma.metrics import binary_metrics
    torch.set_num_threads(1)
    metrics, history = read(run / 'metrics.json'), read(run / 'history.json')
    assert metrics['training_complete'] and metrics['completed_epochs'] == job['epochs']
    assert len(history) == job['epochs'] and [row['epoch'] for row in history] == list(range(job['epochs']))
    best = torch.load(run / 'best.pt', weights_only=True, map_location='cpu')
    last = torch.load(run / 'last.pt', weights_only=True, map_location='cpu')

    def finite(value):
        if isinstance(value, torch.Tensor):
            assert bool(torch.isfinite(value).all())
        elif isinstance(value, dict):
            for item in value.values():
                finite(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                finite(item)

    finite(best)
    finite(last)
    assert best['kind'] == 'inference' and last['kind'] == 'training'
    assert last['epoch'] == job['epochs'] - 1 and last['history'] == history
    assert int(last['ema']['n_averaged']) == last['global_step']
    assert last['optimizer']['state'] and last['scaler'] and last['rng']['cuda'] is not None
    assert best['epoch'] == metrics['best_epoch'] == last['best']['epoch']
    assert all(torch.equal(value, last['best']['model_state'][key]) for key, value in best['model_state'].items())
    del best, last
    with (Path(context['data_root']) / 'valid.csv').open(newline='', encoding='utf-8-sig') as stream:
        expected = list(csv.DictReader(stream))
    with (run / 'valid_predictions.csv').open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    with Path(context['baseline_predictions']).open(newline='', encoding='utf-8') as stream:
        baseline = list(csv.DictReader(stream))
    assert len(rows) == len(expected) == len(baseline) == 5849
    assert [row['Image_name'] for row in rows] == [row['Image_name'] for row in expected] == [row['Image_name'] for row in baseline]
    labels = np.array([int(row['True_Label']) for row in rows])
    assert labels.tolist() == [int(row['True_Label']) for row in expected] and int(labels.sum()) == 191
    probabilities = np.array([float(row['Probability']) for row in rows])
    original = np.array([float(row['Probability']) for row in baseline])
    recomputed = binary_metrics(labels, probabilities)
    for key in KEYS:
        assert abs(recomputed[key] - metrics['valid'][key]) <= 1e-10
    # Paired, stratified bootstrap: identical resampled patients/images for both scores.
    rng = np.random.default_rng(20261023)
    positive, negative = np.flatnonzero(labels), np.flatnonzero(1 - labels)
    deltas = []
    for _ in range(1000):
        selected = np.concatenate((rng.choice(positive, len(positive)), rng.choice(negative, len(negative))))
        deltas.append(roc_auc_score(labels[selected], probabilities[selected]) - roc_auc_score(labels[selected], original[selected]))
    return {'run_directory': str(run), 'completed_epochs': job['epochs'], 'verified': True,
        'best_epoch': metrics['best_epoch'] + 1, 'best_weights_variant': metrics['best_weights_variant'],
        'valid': metrics['valid'], 'delta_from_b0': {key: recomputed[key] - context['baseline_metrics'][key] for key in KEYS},
        'paired_bootstrap_auroc_delta_95ci': np.quantile(deltas, [0.025, 0.975]).tolist(),
        'bootstrap_replicates': 1000, 'bootstrap_seed': 20261023,
        'best_raw_auroc': max(row['valid_raw']['auroc'] for row in history),
        'best_ema_auroc': max(row['valid_ema']['auroc'] for row in history),
        'best_checkpoint': str(run / 'best.pt'), 'last_checkpoint': str(run / 'last.pt'),
        'predictions': str(run / 'valid_predictions.csv'), 'checkpoint_sha256': sha(run / 'best.pt')}


def supervise(root):
    context = read(root / 'context.json')
    state = {'stage': 'cpu_tests', 'jobs': {name: {'stage': 'waiting_for_gpu', **job} for name, job in context['jobs'].items()}}
    save(root / 'status.json', state)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', PYTHONPATH=str(REPO / 'src'))
    with (root / 'tests.log').open('w') as stream:
        test = subprocess.run([PYTHON, '-c', "import torch,unittest;torch.set_num_threads(2);r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover('tests'));raise SystemExit(not r.wasSuccessful())"],
            cwd=REPO, env=environment, stdout=stream, stderr=subprocess.STDOUT)
    (root / 'tests.exit').write_text(str(test.returncode) + '\n')
    assert test.returncode == 0, 'Regression tests failed; training was not started'
    verify_context(context)
    state['stage'] = 'training'
    running = {}
    while any(job['stage'] in ('waiting_for_gpu', 'training') for job in state['jobs'].values()):
        for name, job in state['jobs'].items():
            if job['stage'] == 'waiting_for_gpu':
                try:
                    available = gpu_available(job['gpu'])
                except (subprocess.SubprocessError, ValueError) as exc:
                    job['gpu_check_error'] = str(exc)
                    continue
                if not available:
                    continue
                env = dict(environment, CUDA_VISIBLE_DEVICES=str(job['gpu']))
                log = root / (name + '.log')
                command = [PYTHON, '-u', str(Path(__file__).resolve()), 'worker', str(root), '--job', name]
                with log.open('x') as stream:
                    process = subprocess.Popen(command, cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT)
                running[name] = process
                job.update(stage='training', pid=process.pid,
                    pid_start_ticks=int(Path('/proc', str(process.pid), 'stat').read_text().split()[21]),
                    uid=os.getuid(), cwd=str(REPO), command=command, log=str(log),
                    started_utc=datetime.now(timezone.utc).isoformat())
                print(f"Started {name} on physical GPU{job['gpu']}: {process.pid}", flush=True)
            elif job['stage'] == 'training' and running[name].poll() is not None:
                job['exit_code'] = running[name].returncode
                (root / (name + '.exit')).write_text(str(job['exit_code']) + '\n')
                if job['exit_code']:
                    job['stage'] = 'failed'
                    continue
                run_paths = re.findall(r'^\[run\] (.+)$', Path(job['log']).read_text(), flags=re.M)
                assert len(run_paths) == 1
                job['run_directory'] = run_paths[0]
                job['stage'] = 'verifying'
                save(root / 'status.json', state)
                job['result'] = verify_result(Path(run_paths[0]), context, job)
                save(root / (name + '_result.json'), job['result'])
                job['stage'] = 'complete'
                print(f"Verified {name}: AUROC={job['result']['valid']['auroc']:.9f}", flush=True)
        save(root / 'status.json', state)
        if any(job['stage'] in ('waiting_for_gpu', 'training') for job in state['jobs'].values()):
            time.sleep(30)
    state['stage'] = 'complete' if all(job['stage'] == 'complete' for job in state['jobs'].values()) else 'failed'
    state['completed_utc'] = datetime.now(timezone.utc).isoformat()
    save(root / 'status.json', state)
    save(root / 'summary.json', {'baseline': context['baseline_metrics'], 'jobs': state['jobs'],
        'stage': state['stage'], 'baseline_checkpoint_retained': context['source_snapshot']})
    return 0 if state['stage'] == 'complete' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('worker', 'supervise'))
    parser.add_argument('root', type=Path)
    parser.add_argument('--job')
    parser.add_argument('--resume')
    args = parser.parse_args()
    if args.mode == 'worker':
        worker(args.root, args.job, args.resume)
    else:
        try:
            code = supervise(args.root)
        except BaseException:
            traceback.print_exc()
            code = 1
        (args.root / 'supervisor.exit').write_text(str(code) + '\n')
        raise SystemExit(code)
