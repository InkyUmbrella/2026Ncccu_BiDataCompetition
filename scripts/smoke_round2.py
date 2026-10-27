"""One real-image optimizer update plus a small forward check; not an AUROC experiment."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from glaucoma.config import Config
from glaucoma.data.age import AgeStats
from glaucoma.data.dataset import FundusDataset, read_samples
from glaucoma.data.transforms import build_transform
from glaucoma.engine.evaluation import collect_predictions
from glaucoma.engine.optimization import build_optimizer, train_one_epoch
from glaucoma.models import build_model
from glaucoma.utils import seed_everything


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--data-root',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    mask=os.environ.get('CUDA_VISIBLE_DEVICES')
    if mask not in ('2','3','4') or torch.cuda.device_count()!=1:
        raise RuntimeError('This batch requires exactly one of physical GPUs 2/3/4 to be visible')
    config=Config.load(args.config)
    seed_everything(config.seed)
    torch.set_num_threads(2)
    device=torch.device('cuda:0');torch.cuda.set_device(device)
    rows={split:read_samples(args.data_root,split) for split in ('train','valid')}
    stats=AgeStats.fit(row.age for row in rows['train'])
    loaders={}
    for split,samples in rows.items():
        selected=[row for row in samples if row.label==1][:4]+[row for row in samples if row.label==0][:12]
        transform=build_transform(config.data.image_size,training=split=='train',
            crop_black_border=config.data.crop_black_border,crop_threshold=config.data.crop_threshold,
            crop_margin=config.data.crop_margin,normalization=config.data.normalization)
        dataset=FundusDataset(selected,stats,transform,decode=config.data.decode,image_size=config.data.image_size)
        loaders[split]=DataLoader(dataset,batch_size=16,num_workers=2)
    model=build_model(config.model).to(device)
    first=next(model.encoder.parameters());before=first.detach().clone()
    smoke=replace(config.train,accumulation_steps=1)
    optimizer=build_optimizer(model,smoke)
    scaler=torch.amp.GradScaler('cuda')
    step=0
    torch.cuda.reset_peak_memory_stats(device)
    for attempt in range(4):
        metrics,step=train_one_epoch(model,loaders['train'],optimizer,scaler,device,smoke,step,10,0)
        if step:
            break
    if step!=1 or torch.equal(before,first):
        raise RuntimeError('No successful encoder update in smoke test')
    predictions,_=collect_predictions(model,loaders['valid'],device,smoke.pos_weight,True)
    if len(predictions['probabilities'])!=16:
        raise RuntimeError('Incomplete smoke validation forward')
    report={'physical_gpu':int(mask),'logical_device':str(device),'gpu_name':torch.cuda.get_device_name(device),
            'model':config.model.name,'image_size':config.data.image_size,'weights':config.model.weights,
            'train_rows':16,'valid_rows':16,'optimizer_updates':step,'attempts':attempt+1,
            'encoder_changed':True,'train_loss':metrics['loss'],
            'peak_allocated_MiB':torch.cuda.max_memory_allocated(device)/1024**2,
            'limitation':'Small smoke test only; metrics do not represent a formal validation run.'}
    with Path(args.output).open('x') as handle:
        json.dump(report,handle,indent=2)
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
