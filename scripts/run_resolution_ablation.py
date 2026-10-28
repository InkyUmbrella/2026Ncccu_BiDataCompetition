"""Portable, sequential 512/640 fine-tuning from one inference checkpoint."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

from glaucoma.config import Config
from glaucoma.engine.checkpoint import read_checkpoint
from glaucoma.utils import save_json, sha256_file


def resolve_configs(source, data_root, output, sizes, batch_size=8, workers=4, device='cuda', recipe_family='resolution'):
    if batch_size < 1 or 128 % batch_size:
        raise ValueError('batch-size must be a positive divisor of 128')
    configs={}
    for size in sizes:
        if recipe_family not in ('resolution','ranking'):
            raise ValueError('Unknown experiment family')
        c=Config.load(ROOT/f'configs/experiments/shimmer_{recipe_family}_{size}_seed42.json')
        if source.model != c.model or source.data.image_size != 512:
            raise ValueError('Need the matching 512 ConvNeXt inference checkpoint (best.pt)')
        for field in ('crop_black_border','crop_threshold','crop_margin','decode','normalization'):
            if getattr(source.data,field)!=getattr(c.data,field):
                raise ValueError(f'Checkpoint preprocessing differs: {field}')
        c.data.root=str(Path(data_root).resolve())
        c.data.num_workers=workers
        c.output_root=str(Path(output).resolve()/str(size)/'training')
        c.device=device
        c.train.batch_size=batch_size
        c.train.accumulation_steps=128//batch_size
        configs[size]=c.validate()
    return configs


def run(command, log):
    print('Running:', ' '.join(map(str,command)),flush=True)
    env=dict(os.environ,PYTHONPATH=str(ROOT/'src'),PYTHONUNBUFFERED='1')
    with Path(log).open('x') as handle:
        subprocess.run(list(map(str,command)),cwd=ROOT,env=env,stdout=handle,
                       stderr=subprocess.STDOUT,check=True)


def verify_evaluation(directory, expected):
    with (directory/'predictions.csv').open(newline='') as handle:
        rows=list(csv.DictReader(handle))
    if [(r['Image_name'],int(r['True_Label'])) for r in rows] != expected:
        raise ValueError('Independent validation names/order/labels mismatch')
    return json.loads((directory/'metrics.json').read_text())


def main(args):
    checkpoint=Path(args.checkpoint).resolve()
    payload=read_checkpoint(checkpoint,'inference')
    source=Config.from_dict(payload['config'])
    output=Path(args.output_root).resolve()
    family=getattr(args,'recipe_family','resolution')
    sizes=getattr(args,'recipe_names',None) or ((512,640) if args.sizes=='both' else (int(args.sizes),))
    configs=resolve_configs(source,args.data_root,output,sizes,args.batch_size,args.workers,args.device,family)
    del payload
    with (Path(args.data_root)/'valid.csv').open(encoding='utf-8-sig',newline='') as handle:
        expected=[(r['Image_name'],int(r['True_Label'])) for r in csv.DictReader(handle)]
    hashes={s:sha256_file(Path(args.data_root)/f'{s}.csv') for s in ('train','valid')}
    # A new directory per invocation prevents accidentally overwriting experiments.
    output.mkdir(parents=True,exist_ok=False)
    manifest={'checkpoint':str(checkpoint),'checkpoint_sha256':sha256_file(checkpoint),
              'csv_hashes':hashes,'experiment_family':family,'sizes':list(sizes),'effective_batch':128,
              'status':'prepared','results':[]}
    save_json(output/'manifest.json',manifest)
    for size,c in configs.items():
        (output/str(size)).mkdir()
        save_json(output/str(size)/'config.json',c.to_dict())
    if args.prepare_only:
        print('Prepared:',output);return
    cli=[sys.executable,'-u','-m','glaucoma']
    def evaluate(weights,destination,log):
        run(cli+['evaluate','--checkpoint',weights,'--data-root',args.data_root,
                 '--device',args.device,'--batch-size',args.batch_size,'--num-workers',args.workers,
                 '--output-dir',destination],log)
        return verify_evaluation(destination,expected)
    try:
        baseline=evaluate(checkpoint,output/'baseline512',output/'baseline512.log')
        manifest.update(status='running',baseline512=baseline)
        save_json(output/'manifest.json',manifest)
        for size,c in configs.items():
            directory=output/str(size)
            command=cli+['train','--config',directory/'config.json','--init-from',checkpoint]
            if getattr(args,'image_cache',None):
                command+=['--image-cache',args.image_cache]
            elif args.cache:
                cache=directory/'cache'
                run(cli+['cache-images','--config',directory/'config.json','--data-root',args.data_root,
                         '--output',cache,'--workers',max(1,args.workers),'--storage','png'],directory/'cache.log')
                command+=['--image-cache',cache]
            run(command,directory/'train.log')
            runs=list((directory/'training'/c.experiment).glob('seed*'))
            if len(runs)!=1:raise RuntimeError('Expected exactly one training run')
            training=runs[0]
            metrics=evaluate(training/'best.pt',directory/'evaluation',directory/'evaluate.log')
            saved=json.loads((training/'metrics.json').read_text())
            metadata=json.loads((training/'metadata.json').read_text())
            if metadata['csv_hashes']!=hashes:raise ValueError('Training dataset changed')
            for key in ('auroc','ap','auprc_trapz'):
                if abs(metrics[key]-saved['valid'][key])>1e-8:
                    raise ValueError(f'Independent re-evaluation mismatch: {key}')
            manifest['results'].append({'recipe':str(size),'size':c.data.image_size,'metrics':metrics,
                'initial_metrics':json.loads((training/'initial_validation.json').read_text()),
                'checkpoint':str(training/'best.pt'),'checkpoint_sha256':sha256_file(training/'best.pt'),
                'selected_weights':saved.get('selected_weights'),
                'delta_vs_original512':metrics['auroc']-baseline['auroc']})
            save_json(output/'manifest.json',manifest)
        manifest['status']='complete'
        save_json(output/'manifest.json',manifest)
        print(json.dumps(manifest,indent=2))
    except Exception as error:
        manifest.update(status='failed',error=str(error))
        save_json(output/'manifest.json',manifest)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--data-root',required=True)
    parser.add_argument('--output-root',required=True,help='New output directory')
    parser.add_argument('--sizes',choices=('512','640','both'),default='both')
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--cache',action='store_true',help='Build per-size lossless PNG caches on local SSD')
    parser.add_argument('--prepare-only',action='store_true')
    main(parser.parse_args())
