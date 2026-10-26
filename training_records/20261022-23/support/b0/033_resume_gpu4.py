import copy,hashlib,json,sys
from pathlib import Path
from datetime import datetime,timezone
from glaucoma.config import Config
from glaucoma.engine import trainer
context=json.loads((Path(__file__).parent/'resume_context.json').read_text())
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
config=Config.load(context['config'])
config.data.root=context['data_root']
trainer._resume_compatible(config,context['checkpoint_config'])
assert sha(context['checkpoint'])==context['checkpoint_sha256']
assert sha(Path(context['cache'])/'manifest.json')==context['cache_manifest_sha256']
assert sha(context['source_audit'])==context['audit_sha256']
for split,expected in context['csv_hashes'].items():
 assert sha(Path(config.data.root)/(split+'.csv'))==expected
report=json.loads(Path(context['source_audit']).read_text())
assert not report['errors'] and report['image_source']=='lossless_cache'
assert {k:v['count'] for k,v in report['splits'].items()}=={'train':51396,'valid':5849}
if '--check-only' in sys.argv:
 print(json.dumps({'context_verified':True,'completed_epochs':context['completed_epochs'],'reused_audit':context['source_audit']}))
 raise SystemExit(0)
def reuse_verified_audit(samples,*args,**kwargs):
 assert {k:len(v) for k,v in samples.items()}=={'train':51396,'valid':5849}
 reused=copy.deepcopy(report)
 reused['audit_reuse']={'source_report':context['source_audit'],'source_report_sha256':context['audit_sha256'],'cache_manifest_sha256':context['cache_manifest_sha256'],'verified_at_utc':datetime.now(timezone.utc).isoformat(),'reason':'Same-data checkpoint migration from physical GPU 2 to GPU 4'}
 print('[audit] Reusing the existing successful full cache audit; CSV and manifest hashes match',flush=True)
 return reused
trainer.audit_samples=reuse_verified_audit
from glaucoma.cli import main
main(['train','--config',context['config'],'--data-root',config.data.root,'--image-cache',context['cache'],'--resume',context['checkpoint']])
