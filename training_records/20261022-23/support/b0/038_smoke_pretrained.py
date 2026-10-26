from pathlib import Path
import json
import math
import random
import time

import torch
from torchvision.models import ConvNeXt_Tiny_Weights

from glaucoma.config import Config
from glaucoma.data.age import AgeStats
from glaucoma.data.dataset import FundusDataset, read_samples
from glaucoma.data.transforms import build_transform
from glaucoma.engine.evaluation import collect_predictions
from glaucoma.engine.optimization import build_optimizer, train_one_epoch
from glaucoma.engine.trainer import make_loader
from glaucoma.models import build_model
from glaucoma.utils import seed_everything


torch.set_num_threads(2)
seed_everything(42)
config = Config.load('configs/experiments/b0_crop384_imagenet.json')
config.data.root = '/home/lichunyu/competition/dataset/glaucoma_first'
device = torch.device('cuda:0')
torch.cuda.set_device(device)
started = time.monotonic()
model = build_model(config.model)
official = ConvNeXt_Tiny_Weights.IMAGENET1K_V1.get_state_dict(progress=False, check_hash=True)
matched = 0
for name, tensor in official.items():
    if name.startswith('features.'):
        key = '0.' + name.removeprefix('features.')
    elif name.startswith('classifier.0.'):
        key = '2.' + name.removeprefix('classifier.0.')
    else:
        continue
    torch.testing.assert_close(model.encoder.state_dict()[key], tensor, atol=0, rtol=0)
    matched += 1
del official
model.to(device)
rows = read_samples(config.data.root, 'train')
age_stats = AgeStats.fit(row.age for row in rows)
rng = random.Random(42)
selected = rng.sample([row for row in rows if row.label], 4) + rng.sample([row for row in rows if not row.label], 124)
rng.shuffle(selected)
transform = build_transform(config.data.image_size, training=True,
    crop_black_border=config.data.crop_black_border, crop_threshold=config.data.crop_threshold,
    crop_margin=config.data.crop_margin, normalization=config.data.normalization)
dataset = FundusDataset(selected, age_stats, transform)
# num_workers=0 keeps this bounded test self-contained; the full config uses 8.
loader = make_loader(dataset, config.train.batch_size, 0, 42, training=True, pin_memory=True)
optimizer = build_optimizer(model, config.train)
scaler = torch.amp.GradScaler('cuda')
before = model.encoder[0][0][0].weight.detach().clone()
torch.cuda.reset_peak_memory_stats(device)
batches = math.ceil(len(rows) / config.train.batch_size)
updates = math.ceil(batches / config.train.accumulation_steps)
step, attempts = 0, []
for _ in range(4):
    metrics, step = train_one_epoch(model, loader, optimizer, scaler, device, config.train,
        step, updates * config.train.epochs, updates * config.train.warmup_epochs)
    attempts.append(metrics)
    if step:
        break
assert step == 1, attempts
assert not torch.equal(before, model.encoder[0][0][0].weight)
assert all(torch.isfinite(parameter).all().item() for parameter in model.parameters())
validation = read_samples(config.data.root, 'valid')
sampled_valid = rng.sample([row for row in validation if row.label], 2) + rng.sample([row for row in validation if not row.label], 30)
valid_transform = build_transform(config.data.image_size, crop_black_border=config.data.crop_black_border,
    crop_threshold=config.data.crop_threshold, crop_margin=config.data.crop_margin,
    normalization=config.data.normalization)
valid_loader = make_loader(FundusDataset(sampled_valid, age_stats, valid_transform), 32, 0, 42, pin_memory=True)
predictions, metrics = collect_predictions(model, valid_loader, device, config.train.pos_weight, True)
assert metrics['count'] == 32
report = {'gpu': torch.cuda.get_device_name(device), 'physical_gpu': 2, 'torch': str(torch.__version__),
    'weights': config.model.weights, 'matched_encoder_tensors': matched,
    'image_size': config.data.image_size, 'normalization': config.data.normalization,
    'batch_size': config.train.batch_size, 'accumulation_steps': config.train.accumulation_steps,
    'training_images': len(selected), 'training_positives': sum(row.label for row in selected),
    'optimizer_steps': step, 'attempts': attempts, 'stem_changed': True, 'all_parameters_finite': True,
    'validation_images': metrics['count'], 'peak_allocated_mib': round(torch.cuda.max_memory_allocated() / 1024**2, 1),
    'seconds': round(time.monotonic() - started, 2),
    'note': 'Bounded engineering smoke test only; not a full validation AUROC/AP result.'}
path = Path('outputs/pretrained-validation/cuda-real-images.json')
path.write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2), flush=True)
