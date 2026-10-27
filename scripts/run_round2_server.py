"""Run the approved GPU 2/3/4 experiment batch, with caches and automatic evaluation.

The source checkout stays unchanged. All logs, resolved configs, checkpoints,
verification reports and a small JSON-only archive live under --work-root.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import json
import os
import random
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
CONFIGS = ROOT / 'configs/experiments'
PYTHON = sys.executable
LOCK = threading.Lock()
STATE = {'phase': 'preparing', 'jobs': {}, 'finished': False}


def stamp():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def state(**values):
    with LOCK:
        STATE.update(values)
        STATE['updated_at'] = stamp()
        save(WORK / 'state.json', STATE)


def job_state(name, **values):
    with LOCK:
        STATE['jobs'].setdefault(name, {}).update(values)
        STATE['updated_at'] = stamp()
        save(WORK / 'state.json', STATE)


def environment(gpu=''):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(ROOT / 'src'),
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', TORCH_HOME=str(WORK / 'torch'),
               PYTHONUNBUFFERED='1')
    return env


def run(command, log, gpu='', name=None):
    with Path(log).open('ab') as handle:
        process = subprocess.Popen(command, cwd=ROOT, env=environment(gpu),
                                   stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        if name:
            job_state(name, pid=process.pid, command=command, log=str(log), gpu=gpu)
        code = process.wait()
    if code:
        raise RuntimeError(f'Command exited {code}; see {log}')


def free_gpu(index, timeout=120):
    if index not in (2, 3, 4):
        raise ValueError('This launch is restricted to physical GPUs 2, 3 and 4')
    deadline = time.monotonic() + timeout
    while True:
        output = subprocess.check_output(['nvidia-smi', '-i', str(index),
            '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
        if not output.strip():
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(f'GPU {index} is still occupied; refusing to start: {output.strip()}')
        print(f'[wait] GPU {index} has compute processes; waiting without interrupting them', flush=True)
        time.sleep(30)


def prepare_cache(size):
    config = ('shimmer_512_pretrained_reference.json' if size == 512
              else 'shimmer_efficientnetv2s_384_reference_seed42.json')
    destination = CACHE_ROOT / f'cache{size}'
    run([PYTHON, '-m', 'glaucoma', 'cache-images', '--config', str(CONFIGS / config),
         '--data-root', DATA, '--output', str(destination), '--workers', '4', '--storage', CACHE_STORAGE],
         WORK / f'cache{size}.log')
    return str(destination)


def budget_png_cache():
    from glaucoma.data.dataset import read_samples, load_rgb
    from glaucoma.data.transforms import build_preprocess
    samples = [row for split in ('train','valid') for row in read_samples(DATA, split)]
    selected = random.Random(42).sample(samples, min(64, len(samples)))
    estimates = {}
    for size in (512,384):
        lengths = []
        transform = build_preprocess(size, True, 10, .03)
        for sample in selected:
            image = transform(load_rgb(sample.path, 'draft', size))
            buffer = io.BytesIO()
            image.save(buffer, format='PNG', compress_level=1)
            lengths.append(buffer.tell())
        estimates[str(size)] = sum(lengths) / len(lengths) * len(samples)
    required = sum(estimates.values()) * 1.25 + 512 * 1024**2
    available = shutil.disk_usage(CACHE_ROOT).free
    memory = {line.split(':')[0]: int(line.split()[1]) * 1024
              for line in Path('/proc/meminfo').read_text().splitlines()}
    # Only reserve RAM for tmpfs. A disk PNG cache can instead use reclaimable page cache.
    in_memory = str(CACHE_ROOT).startswith(('/dev/shm/', '/tmp/'))
    reserve = 12 * 1024**3
    if required > available or (in_memory and required + reserve > memory['MemAvailable']):
        raise RuntimeError('Compressed cache budget exceeds storage/RAM headroom; no cache or training started')
    state(cache_budget={'estimated_bytes':estimates, 'required_with_headroom':required,
                        'available_storage':available, 'available_memory':memory['MemAvailable'],
                        'memory_reserve_bytes':reserve, 'in_memory':in_memory})


def experiment(name, config_name, gpu, cache, init_from=None):
    directory = WORK / 'jobs' / name
    directory.mkdir(parents=True, exist_ok=False)
    config = json.loads((CONFIGS / config_name).read_text())
    config['data']['root'] = DATA
    config['output_root'] = str(directory / 'training')
    config['device'] = 'cuda:0'
    resolved = directory / 'resolved_config.json'
    save(resolved, config)
    command = [PYTHON, '-u', '-m', 'glaucoma', 'train', '--config', str(resolved), '--image-cache', cache]
    if init_from:
        command += ['--init-from', str(init_from)]
    job_state(name, status='waiting_gpu', gpu=gpu, waiting_since=stamp())
    free_gpu(gpu, timeout=24*3600)
    job_state(name, status='training', started_at=stamp(), config=str(resolved), cache=cache)
    run(command, directory / 'train.log', gpu, name)
    runs = list((directory / 'training' / config['experiment']).glob('seed*'))
    if len(runs) != 1:
        raise RuntimeError(f'Expected exactly one run for {name}, got {runs}')
    training = runs[0]
    job_state(name, status='evaluating', run_dir=str(training))
    evaluation = directory / 'evaluation'
    run([PYTHON, '-u', '-m', 'glaucoma', 'evaluate', '--checkpoint', str(training / 'best.pt'),
         '--data-root', DATA, '--device', 'cuda:0', '--batch-size', str(config['train']['batch_size']),
         '--num-workers', '4', '--output-dir', str(evaluation)], directory / 'evaluate.log', gpu, name)
    metrics = json.loads((evaluation / 'metrics.json').read_text())
    original = json.loads((training / 'metrics.json').read_text())
    metadata = json.loads((training / 'metadata.json').read_text())
    if metadata['csv_hashes'] != STATE['csv_hashes']:
        raise RuntimeError('Training CSV hashes do not match the batch manifest')
    if metrics['count'] != 5849 or metrics['positives'] != 191:
        raise RuntimeError('Unexpected validation row/positive count')
    for key in ('auroc', 'ap', 'auprc_trapz'):
        if abs(metrics[key] - original['valid'][key]) > 1e-8:
            raise RuntimeError(f'Independent evaluation mismatch for {key}')
    import csv
    with (evaluation / 'predictions.csv').open(newline='') as handle:
        predictions = list(csv.DictReader(handle))
    with (Path(DATA) / 'valid.csv').open(encoding='utf-8-sig', newline='') as handle:
        expected = list(csv.DictReader(handle))
    if [row['Image_name'] for row in predictions] != [row['Image_name'] for row in expected]:
        raise RuntimeError('Prediction order/name mismatch')
    if [int(row['True_Label']) for row in predictions] != [int(row['True_Label']) for row in expected]:
        raise RuntimeError('Prediction labels do not match official valid.csv')
    archive = WORK / 'archive' / name
    archive.mkdir(parents=True, exist_ok=False)
    for filename in ('config.json', 'metadata.json', 'history.json', 'metrics.json', 'initial_validation.json'):
        source = training / filename
        if source.exists():
            shutil.copy2(source, archive / filename)
    shutil.copy2(evaluation / 'metrics.json', archive / 'reevaluation.json')
    artifact = {'path': str(training / 'best.pt'), 'sha256': sha(training / 'best.pt'),
                'bytes': (training / 'best.pt').stat().st_size,
                'validation_predictions_sha256': sha(evaluation / 'predictions.csv')}
    save(archive / 'artifacts.json', artifact)
    result = {'name': name, 'gpu': gpu, 'metrics': metrics, 'run_dir': str(training),
              'artifact': artifact, 'best_epoch': original['best_epoch'],
              'selected_weights': original.get('selected_weights'), 'ended_at': stamp()}
    job_state(name, status='complete', result=result)
    return training / 'best.pt'


def branch(kind, gpu):
    name = kind
    try:
        config_name = {'seed42':'shimmer_512_pretrained_reference.json',
                       'seed43':'shimmer_512_pretrained_reference_seed43.json',
                       'efficientnet':'shimmer_efficientnetv2s_384_reference_seed42.json'}[kind]
        job_state(name, status='waiting_gpu', gpu=gpu, waiting_since=stamp())
        free_gpu(gpu, timeout=24*3600)
        job_state(name, status='smoke_test', gpu=gpu)
        run([PYTHON, str(ROOT / 'scripts/smoke_round2.py'), '--config', str(CONFIGS / config_name),
             '--data-root', DATA, '--output', str(WORK / f'cuda-smoke-{gpu}.json')],
             WORK / f'cuda-smoke-{gpu}.log', gpu)
        if kind == 'seed42':
            checkpoint = experiment(name, 'shimmer_512_pretrained_reference.json', gpu, CACHE512)
            name = 'seed42_refine'
            experiment(name, 'shimmer_512_reference_refine_seed42.json', gpu, CACHE512, checkpoint)
        elif kind == 'seed43':
            experiment(name, 'shimmer_512_pretrained_reference_seed43.json', gpu, CACHE512)
        else:
            experiment(name, 'shimmer_efficientnetv2s_384_reference_seed42.json', gpu, CACHE384)
    except Exception:
        job_state(name, status='failed', error=traceback.format_exc())


def main():
    state(started_at=stamp(), source_revision=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        gpus=[2,3,4], data_root=DATA,
        csv_hashes={split: sha(Path(DATA) / f'{split}.csv') for split in ('train','valid')})
    if CACHE_STORAGE == 'npy' and shutil.disk_usage(CACHE_ROOT).free < 95 * 1024 ** 3:
        raise RuntimeError('Need at least 95 GiB free before building both caches and training')
    if CACHE_STORAGE == 'png':
        budget_png_cache()
    checkpoint_dir = WORK / 'torch/hub/checkpoints'
    checkpoint_dir.mkdir(parents=True)
    source = Path('/home/lichunyu/.cache/torch/hub/checkpoints/convnext_tiny-983f1562.pth')
    if sha(source) != '983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d':
        raise RuntimeError('Existing official ConvNeXt weights failed SHA256 verification')
    shutil.copy2(source, checkpoint_dir / source.name)
    if EFFICIENTNET_WEIGHTS:
        supplied = Path(EFFICIENTNET_WEIGHTS)
        if sha(supplied) != 'dd5fe13b1d60ec15317ccc8ca158186e134d3366c3dde9cb9a4e301f2dc66c74':
            raise RuntimeError('Supplied official EfficientNet weights failed SHA256 verification')
        shutil.copy2(supplied, checkpoint_dir / 'efficientnet_v2_s-dd5fe13b.pth')
    run([PYTHON, '-c', 'from torchvision.models import EfficientNet_V2_S_Weights; '
         'EfficientNet_V2_S_Weights.IMAGENET1K_V1.get_state_dict(progress=True,check_hash=True)'], WORK / 'weights.log')
    state(pretrained_files={p.name: {'sha256': sha(p), 'bytes': p.stat().st_size}
                           for p in checkpoint_dir.glob('*.pth')})
    global CACHE512, CACHE384
    state(phase='caching')
    with ThreadPoolExecutor(max_workers=2) as pool:
        first=pool.submit(prepare_cache,512);second=pool.submit(prepare_cache,384)
        CACHE512,CACHE384=first.result(),second.result()
    state(phase='running', caches={'512':CACHE512,'384':CACHE384})
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(lambda pair: branch(*pair), [('seed42',2),('seed43',3),('efficientnet',4)]))
    results = [job['result'] for job in STATE['jobs'].values() if job.get('status') == 'complete']
    ordered = sorted(results,key=lambda x:(x['metrics']['auroc'],x['metrics']['auprc_trapz']),reverse=True)
    summary = {'started_at':STATE['started_at'],'ended_at':stamp(),'source_revision':STATE['source_revision'],
               'csv_hashes':STATE['csv_hashes'],'results':ordered,
               'previous_reported_auroc':0.9885271098329013,
               'previous_weights_available':False,
               'winner':ordered[0]['name'] if ordered else None,
               'failed_jobs':{k:v['error'] for k,v in STATE['jobs'].items() if v.get('status')=='failed'}}
    comparisons=[]
    for candidate in ('seed42_refine','seed43','efficientnet'):
        if STATE['jobs'].get(candidate,{}).get('status')=='complete' and STATE['jobs'].get('seed42',{}).get('status')=='complete':
            new_run=Path(STATE['jobs'][candidate]['result']['run_dir'])
            old_run=Path(STATE['jobs']['seed42']['result']['run_dir'])
            log=WORK / 'archive' / f'compare_{candidate}_vs_seed42.log'
            try:
                run([PYTHON,str(ROOT / 'scripts/compare_predictions.py'),
                     str(new_run/'valid_predictions.csv'),str(old_run/'valid_predictions.csv'),'--reps','1000'],log)
                comparisons.append(str(log.name))
            except Exception:
                comparisons.append({'failed':candidate,'error':traceback.format_exc()})
    summary['comparison_reports']=comparisons
    save(WORK / 'archive/summary.json',summary)
    state(phase='finished',finished=True,summary=summary)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',required=True)
    parser.add_argument('--work-root',required=True,help='New private directory on local SSD (at least 95 GiB free)')
    parser.add_argument('--efficientnet-weights',help='Optional verified official file, to avoid a slow server download')
    parser.add_argument('--cache-root',help='New separate cache directory, for example on a private memory disk')
    parser.add_argument('--cache-storage',choices=('npy','png'),default='npy')
    args=parser.parse_args()
    os.umask(0o077)
    DATA=str(Path(args.data_root).resolve())
    EFFICIENTNET_WEIGHTS=args.efficientnet_weights
    WORK=Path(args.work_root).resolve()
    WORK.mkdir(parents=True,exist_ok=False)
    CACHE_ROOT=Path(args.cache_root).resolve() if args.cache_root else WORK
    CACHE_STORAGE=args.cache_storage
    if CACHE_ROOT != WORK:
        CACHE_ROOT.mkdir(parents=True,exist_ok=False)
    try:
        main()
    except Exception:
        state(phase='failed',finished=True,error=traceback.format_exc())
        raise
