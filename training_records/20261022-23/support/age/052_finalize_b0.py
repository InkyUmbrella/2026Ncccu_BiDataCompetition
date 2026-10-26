"""Verify the completed original B0 and its evaluation in the merged worktree."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone

import torch

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parent
MERGED = Path('/home/lichunyu/competition-fwq-pretrained')
ORIGINAL = Path('/home/lichunyu/competition-pretrained')
KEYS = ('auroc', 'ap', 'auprc_trapz')


def save(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2), encoding='utf-8')


def stage(value):
    (ROOT / 'stage').write_text(value + '\n', encoding='utf-8')
    print(value, flush=True)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def finite_tensors(value):
    if isinstance(value, torch.Tensor):
        assert bool(torch.isfinite(value).all()), 'Non-finite checkpoint tensor'
    elif isinstance(value, dict):
        for item in value.values():
            finite_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            finite_tensors(item)


def predictions(metrics, path, data):
    from glaucoma.metrics import binary_metrics
    with (data / 'valid.csv').open(newline='', encoding='utf-8-sig') as stream:
        expected = list(csv.DictReader(stream))
    with Path(path).open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(expected) == 5849
    assert [r['Image_name'] for r in rows] == [r['Image_name'] for r in expected]
    labels = [int(r['True_Label']) for r in rows]
    assert labels == [int(r['True_Label']) for r in expected]
    assert sum(labels) == 191 and len({r['Image_name'] for r in rows}) == 5849
    probabilities = [float(r['Probability']) for r in rows]
    assert all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities)
    assert all(math.isfinite(float(r['Logit'])) for r in rows)
    recomputed = binary_metrics(labels, probabilities)
    assert metrics['count'] == 5849 and metrics['positives'] == 191
    for key in KEYS:
        assert math.isfinite(metrics[key]) and abs(metrics[key] - recomputed[key]) <= 1e-10
    return {key: recomputed[key] for key in KEYS}


def main():
    from glaucoma.config import Config
    stage('b0_artifact_verification')
    launch = json.loads((ORIGINAL / 'outputs/pretrained-validation/launch.json').read_text())
    transfer = json.loads((MERGED / 'outputs/fwq-transfer/launch.json').read_text())
    assert Path(launch['exit_file']).read_text().strip() == '0'
    # A reused PID must never be mistaken for the completed original trainer.
    proc = Path('/proc/2268394/stat')
    if proc.exists():
        assert proc.read_text().split()[21] != '131500283', 'Original trainer is still alive'
    run = Path(launch['run_directory'])
    config = json.loads((run / 'config.json').read_text())
    metrics = json.loads((run / 'metrics.json').read_text())
    history = json.loads((run / 'history.json').read_text())
    assert metrics['training_complete'] is True and metrics['completed_epochs'] == 30
    assert len(history) == 30 and [r['epoch'] for r in history] == list(range(30))
    assert config['train']['epochs'] == 30
    assert config['train']['batch_size'] * config['train']['accumulation_steps'] == 128
    assert config['data']['image_size'] == 384 and config['data']['normalization'] == 'imagenet'
    best = torch.load(run / 'best.pt', map_location='cpu', weights_only=True)
    last = torch.load(run / 'last.pt', map_location='cpu', weights_only=True)
    assert best['kind'] == 'inference' and last['kind'] == 'training'
    assert last['epoch'] == 29 and last['history'] == history
    canonical = lambda value: Config.from_dict(value).to_dict()
    assert canonical(last['config']) == canonical(best['config']) == canonical(config)
    assert last['global_step'] == history[-1]['global_step']
    assert all(key in last for key in ('optimizer', 'scaler', 'rng', 'best', 'csv_hashes'))
    assert last['optimizer']['state'] and last['rng']['cuda'] is not None
    finite_tensors(last)
    finite_tensors(best)
    assert best['epoch'] == metrics['best_epoch'] == last['best']['epoch']
    assert best['model_state'].keys() == last['best']['model_state'].keys()
    assert all(torch.equal(value, last['best']['model_state'][key]) for key, value in best['model_state'].items())
    for row in history:
        assert row['valid']['count'] == 5849 and row['valid']['positives'] == 191
        assert all(math.isfinite(row['valid'][key]) for key in KEYS)
    data = Path(config['data']['root'])
    assert all(sha(data / (split + '.csv')) == digest for split, digest in last['csv_hashes'].items())
    original_scores = predictions(metrics['valid'], run / 'valid_predictions.csv', data)
    for key in KEYS:
        assert abs(original_scores[key] - best['metrics'][key]) <= 1e-10
    fixed = Path(transfer['source_snapshot'])
    assert sha(fixed) == transfer['source_sha256']
    fixed_payload = torch.load(fixed, map_location='cpu', weights_only=True)
    source_sha = sha(run / 'best.pt')
    final_snapshot = ROOT / 'final_b0_snapshot.pt'
    assert not final_snapshot.exists()
    shutil.copyfile(run / 'best.pt', final_snapshot)
    assert sha(final_snapshot) == source_sha
    same_model = best['model_state'].keys() == fixed_payload['model_state'].keys() and all(
        torch.equal(value, fixed_payload['model_state'][key]) for key, value in best['model_state'].items())
    report = {
        'verified_at_utc': datetime.now(timezone.utc).isoformat(),
        'training_complete': True, 'completed_epochs': 30, 'original_exit_code': 0,
        'run': str(run), 'best_epoch': best['epoch'] + 1,
        'best_checkpoint': str(run / 'best.pt'), 'last_checkpoint': str(run / 'last.pt'),
        'valid_predictions': str(run / 'valid_predictions.csv'), 'valid': metrics['valid'],
        'checkpoints_verified_readable': True, 'checkpoint_tensors_finite': True,
        'prediction_order_labels_verified': True, 'global_metrics_recomputed': original_scores,
        'snapshot': str(final_snapshot), 'sha256': source_sha,
        'age_initialization_snapshot': str(fixed), 'age_initialization_sha256': transfer['source_sha256'],
        'file_sha256_equal_to_age_snapshot': source_sha == transfer['source_sha256'],
        'model_tensors_equal_to_age_snapshot': same_model,
        'device_migrations': 1, 'fault_recoveries': 0,
    }
    save('original_b0_completion.json', report)
    del last, best, fixed_payload
    if source_sha == transfer['source_sha256']:
        comparison = json.loads((Path(transfer['run_root']) / 'b0_comparison.json').read_text())
        assert comparison['matched'] is True
        output = Path(transfer['b0_evaluation'])
        report['evaluation_reused'] = True
    else:
        stage('gpu_check_before_final_b0')
        guard = subprocess.run([sys.executable, str(Path(transfer['run_root']) / 'guard_gpu.py'), '4'],
                               text=True, capture_output=True)
        (ROOT / 'gpu_check.log').write_text(guard.stdout + guard.stderr)
        (ROOT / 'gpu_check.exit').write_text(str(guard.returncode) + '\n')
        if guard.returncode:
            stage('waiting_for_gpu4')
            return 75
        stage('final_b0_evaluation')
        output = ROOT / 'evaluation'
        command = [sys.executable, '-u', '-m', 'glaucoma', 'evaluate', '--checkpoint', str(final_snapshot),
                   '--data-root', str(data), '--device', 'cuda:0', '--batch-size', '32', '--num-workers', '4',
                   '--split', 'valid', '--output-dir', str(output)]
        save('evaluation_command.json', {'argv': command, 'cwd': str(MERGED), 'physical_gpu': 4})
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES='4', OMP_NUM_THREADS='2', PYTHONPATH=str(MERGED / 'src'))
        with (ROOT / 'evaluation.log').open('w') as stream:
            result = subprocess.run(command, cwd=MERGED, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        (ROOT / 'evaluation.exit').write_text(str(result.returncode) + '\n')
        assert result.returncode == 0, 'Final B0 evaluate failed; preserve its output and log'
        report['evaluation_reused'] = False
    stage('final_b0_verification')
    evaluated = json.loads((output / 'metrics.json').read_text())
    predictions(evaluated, output / 'predictions.csv', data)
    assert sha(final_snapshot) == source_sha and sha(fixed) == transfer['source_sha256']
    differences = {key: evaluated[key] - original_scores[key] for key in KEYS}
    assert all(abs(value) <= 1e-4 for value in differences.values()), differences
    report.update(merged_valid=evaluated, differences=differences, tolerance=1e-4,
                  merged_validation_verified=True, evaluation_output=str(output),
                  merged_predictions=str(output / 'predictions.csv'))
    save('final_b0_comparison.json', report)
    stage('complete')
    return 0


try:
    code = main()
except BaseException:
    traceback.print_exc()
    stage('failed')
    code = 1
(ROOT / 'finalization.exit').write_text(str(code) + '\n')
raise SystemExit(code)
