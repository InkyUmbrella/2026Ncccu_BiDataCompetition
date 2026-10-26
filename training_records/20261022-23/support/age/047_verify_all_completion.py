"""CPU-only artifact checks after both authorized training jobs have exited."""
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import traceback

import torch
from glaucoma.config import Config
from glaucoma.metrics import binary_metrics

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parent
REPO = Path('/home/lichunyu/competition-fwq-pretrained')
KEYS = ('auroc', 'ap', 'auprc_trapz')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def stage(value):
    (ROOT / 'stage').write_text(value + '\n')
    print(value, flush=True)


def finite(value):
    if isinstance(value, torch.Tensor):
        assert bool(torch.isfinite(value).all()), 'Non-finite checkpoint tensor'
    elif isinstance(value, dict):
        for item in value.values():
            finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            finite(item)


def main():
    stage('verifying_completion')
    launch = read(REPO / 'outputs/fwq-transfer/launch.json')
    transfer = Path(launch['run_root'])
    for path in (launch['exit_file'], transfer / 'age_training.exit', transfer / 'age_verification.exit'):
        assert Path(path).read_text().strip() == '0', str(path)
    assert Path(launch['stage_file']).read_text().strip() == 'complete'
    proc = Path('/proc/2264815/stat')
    if proc.exists():
        assert proc.read_text().split()[21] != '131333259', 'Age trainer is still alive'
    final_launch = read(REPO / 'outputs/fwq-transfer/final_b0_launch.json')
    assert Path(final_launch['exit_file']).read_text().strip() == '0'
    final_b0 = read(Path(final_launch['run_root']) / 'final_b0_comparison.json')
    assert final_b0['completed_epochs'] == 30 and final_b0['original_exit_code'] == 0
    assert final_b0['checkpoints_verified_readable'] and final_b0['checkpoint_tensors_finite']
    assert final_b0['merged_validation_verified'] and final_b0['prediction_order_labels_verified']
    assert sha(final_b0['best_checkpoint']) == final_b0['sha256'] == launch['source_sha256']
    assert sha(launch['source_snapshot']) == launch['source_sha256']
    summary = read(transfer / 'age_validation_summary.json')
    run = Path(summary['run'])
    assert run.parent.parent == Path(launch['age_output_root'])
    config, metadata = read(run / 'config.json'), read(run / 'metadata.json')
    history, metrics = read(run / 'history.json'), read(run / 'metrics.json')
    assert metadata['initialized_from'] == launch['source_snapshot']
    assert metadata['resumed_from'] is None
    assert metrics['training_complete'] is True and metrics['completed_epochs'] == 20
    assert len(history) == 20 and [row['epoch'] for row in history] == list(range(20))
    assert config['train']['epochs'] == 20
    assert config['train']['batch_size'] == 64 and config['train']['accumulation_steps'] == 2
    assert config['train']['freeze_image_epochs'] == 3 and config['train']['image_lr_scale'] == 0.1
    assert config['data']['image_size'] == 384 and config['data']['normalization'] == 'imagenet'
    assert config['model']['name'] == 'convnext_tiny_age_residual'
    assert [row['image_frozen'] for row in history] == [True] * 3 + [False] * 17
    for row in history:
        assert row['valid']['count'] == 5849 and row['valid']['positives'] == 191
        assert all(math.isfinite(row['valid'][key]) for key in KEYS)
    stage('checking_age_checkpoints')
    best = torch.load(run / 'best.pt', map_location='cpu', weights_only=True)
    last = torch.load(run / 'last.pt', map_location='cpu', weights_only=True)
    assert best['kind'] == 'inference' and last['kind'] == 'training'
    assert best['epoch'] == metrics['best_epoch'] == last['best']['epoch']
    selected = max(history, key=lambda row: (row['valid']['auroc'], row['valid']['auprc_trapz']))
    assert best['epoch'] == selected['epoch']
    assert last['epoch'] == 19 and last['history'] == history
    assert last['global_step'] == history[-1]['global_step']
    skipped = sum(row['train']['skipped_optimizer_steps'] for row in history)
    assert last['global_step'] == 20 * 402 - skipped
    canonical = lambda value: Config.from_dict(value).to_dict()
    assert canonical(best['config']) == canonical(last['config']) == canonical(config)
    assert last['optimizer']['state'] and last['optimizer']['param_groups']
    assert last['scaler'] and all(key in last['rng'] for key in ('python', 'numpy', 'torch', 'cuda'))
    assert last['rng']['cuda'] is not None
    assert best['model_state'].keys() == last['best']['model_state'].keys()
    assert all(torch.equal(value, last['best']['model_state'][key]) for key, value in best['model_state'].items())
    finite(best)
    finite(last)
    branch = {}
    for name, checkpoint in (('best', best), ('last', last)):
        weight = checkpoint['model_state']['age_residual.2.weight']
        bias = checkpoint['model_state']['age_residual.2.bias']
        assert int(torch.count_nonzero(weight)) > 0, 'Zero-initialized age output branch did not update'
        branch[name] = {'weight_nonzero': int(torch.count_nonzero(weight)),
                        'weight_l2': float(weight.norm()), 'bias': bias.tolist()}
    data = Path(config['data']['root'])
    assert all(sha(data / (split + '.csv')) == digest for split, digest in last['csv_hashes'].items())
    assert metadata['csv_hashes'] == last['csv_hashes']
    stage('checking_predictions')
    with (data / 'valid.csv').open(newline='', encoding='utf-8-sig') as stream:
        expected = list(csv.DictReader(stream))
    prediction_path = run / 'valid_predictions.csv'
    with prediction_path.open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    assert len(expected) == len(rows) == 5849
    assert [row['Image_name'] for row in rows] == [row['Image_name'] for row in expected]
    assert len({row['Image_name'] for row in rows}) == 5849
    labels = [int(row['True_Label']) for row in rows]
    assert labels == [int(row['True_Label']) for row in expected] and sum(labels) == 191
    probabilities = [float(row['Probability']) for row in rows]
    assert all(math.isfinite(value) and 0 <= value <= 1 for value in probabilities)
    assert all(math.isfinite(float(row['Logit'])) for row in rows)
    recomputed = binary_metrics(labels, probabilities)
    scores = {key: recomputed[key] for key in KEYS}
    for key in KEYS:
        assert abs(scores[key] - metrics['valid'][key]) <= 1e-10
        assert abs(scores[key] - best['metrics'][key]) <= 1e-10
        assert abs(scores[key] - summary['valid'][key]) <= 1e-10
    assert metrics['valid']['count'] == 5849 and metrics['valid']['positives'] == 191
    report = {
        'verified_at_utc': datetime.now(timezone.utc).isoformat(), 'all_tasks_complete': True,
        'snapshot_b0': {'best_epoch': launch['source_best_epoch'], 'valid': launch['source_metrics'],
                        'checkpoint': launch['source_snapshot'], 'sha256': launch['source_sha256'],
                        'predictions': final_b0['merged_predictions'], 'log': launch['b0_log']},
        'final_b0': final_b0,
        'age': {'completed_epochs': 20, 'training_complete': True, 'training_exit_code': 0,
                'pipeline_exit_code': 0, 'best_epoch': best['epoch'] + 1, 'last_epoch': 20,
                'valid': metrics['valid'], 'recomputed_global_metrics': scores,
                'delta_from_b0': {key: scores[key] - final_b0['valid'][key] for key in KEYS},
                'best_checkpoint': str(run / 'best.pt'), 'last_checkpoint': str(run / 'last.pt'),
                'best_sha256': sha(run / 'best.pt'), 'last_sha256': sha(run / 'last.pt'),
                'valid_predictions': str(prediction_path), 'log': launch['age_log'],
                'initialization_checkpoint': launch['source_snapshot'],
                'initialization_sha256': launch['source_sha256'],
                'checkpoint_tensors_finite': True, 'optimizer_scaler_rng_history_verified': True,
                'prediction_order_labels_verified': True, 'age_branch': branch,
                'global_step': last['global_step'], 'amp_skipped_updates': skipped,
                'fault_recoveries': 0},
        'fault_recoveries': {'b0': 0, 'age': 0}, 'user_requested_device_migrations': 1,
        'integration_tests_exit_code': int((REPO / 'outputs/fwq-transfer/integration-tests.exit').read_text()),
        'verification_script': str(Path(__file__).resolve()),
    }
    assert report['integration_tests_exit_code'] == 0
    destination = ROOT / 'final_completion_verification.json'
    with destination.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    stage('complete')
    return 0


try:
    code = main()
except BaseException:
    traceback.print_exc()
    stage('failed')
    code = 1
(ROOT / 'verification.exit').write_text(str(code) + '\n')
raise SystemExit(code)
