import csv,json,math,sys
from pathlib import Path
import torch
torch.set_num_threads(1)
root=Path(__file__).resolve().parent
snapshot=json.loads((root/'snapshot.json').read_text())
keys=('auroc','ap','auprc_trapz')
data=Path(snapshot['data_root'])
with (data/'valid.csv').open(newline='',encoding='utf-8-sig') as h: expected=list(csv.DictReader(h))
def validate(metrics,path):
 with Path(path).open(newline='',encoding='utf-8') as h: rows=list(csv.DictReader(h))
 assert len(rows)==len(expected)==5849,'Unexpected validation row count'
 assert [r['Image_name'] for r in rows]==[r['Image_name'] for r in expected],'Validation order/names differ'
 assert [int(r['True_Label']) for r in rows]==[int(r['True_Label']) for r in expected],'Labels differ'
 assert len({r['Image_name'] for r in rows})==5849
 assert sum(int(r['True_Label']) for r in rows)==191
 assert all(math.isfinite(float(r['Probability'])) and 0<=float(r['Probability'])<=1 and math.isfinite(float(r['Logit'])) for r in rows)
 assert metrics['count']==5849 and metrics['positives']==191
 assert all(math.isfinite(metrics[k]) and 0<=metrics[k]<=1 for k in keys)
 return rows
if sys.argv[1]=='b0':
 output=root/'b0_evaluation';metrics=json.loads((output/'metrics.json').read_text());validate(metrics,output/'predictions.csv')
 differences={k:metrics[k]-snapshot['metrics'][k] for k in keys}
 report={'valid':metrics,'snapshot_metrics':snapshot['metrics'],'differences':differences,'tolerance':1e-4,'source_sha256':snapshot['sha256'],'matched':all(abs(v)<=1e-4 for v in differences.values())}
 (root/'b0_comparison.json').write_text(json.dumps(report,indent=2))
 assert report['matched'],'B0 migration changed validation metrics; inspect before age training'
 print(json.dumps({k:metrics[k] for k in keys}))
elif sys.argv[1]=='age':
 runs=sorted((root/'age_training'/'b0_crop384_age_residual_imagenet').glob('seed*'));assert len(runs)==1,'Expected one age training run'
 run=runs[0];metrics=json.loads((run/'metrics.json').read_text());assert metrics['training_complete'] is True and metrics['completed_epochs']==20
 validate(metrics['valid'],run/'valid_predictions.csv')
 best=torch.load(run/'best.pt',map_location='cpu',weights_only=True);last=torch.load(run/'last.pt',map_location='cpu',weights_only=True)
 assert best['kind']=='inference' and last['kind']=='training' and last['epoch']==19 and len(last['history'])==20
 assert all(k in last for k in ('optimizer','scaler','rng','best'))
 assert best['config']['model']['name']=='convnext_tiny_age_residual' and best['config']['data']['normalization']=='imagenet'
 assert all(bool(torch.isfinite(v).all()) for v in best['model_state'].values())
 assert bool(best['model_state']['age_residual.2.weight'].abs().sum()>0),'Age branch did not learn'
 b0=json.loads((root/'b0_evaluation'/'metrics.json').read_text())
 report={'run':str(run),'best_epoch':best['epoch']+1,'completed_epochs':20,'valid':metrics['valid'],'delta_from_snapshot_b0':{k:metrics['valid'][k]-b0[k] for k in keys},'best_checkpoint':str(run/'best.pt'),'valid_predictions':str(run/'valid_predictions.csv'),'initialized_from_sha256':snapshot['sha256'],'checkpoints_verified_readable':True}
 (root/'age_validation_summary.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:metrics['valid'][k] for k in keys}))
else:raise ValueError('Unknown validation phase')
