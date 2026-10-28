import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch

from test_baseline import fixture
from test_fusion import load_script
from glaucoma.cli import main
from glaucoma.config import Config, ModelConfig
from glaucoma.engine.checkpoint import read_checkpoint
from glaucoma.engine.optimization import pairwise_rank_loss
from glaucoma.engine.trainer import train


class RankingTests(unittest.TestCase):
    def test_ranking_gradient_offset_and_missing_classes(self):
        logits=torch.tensor([-2.,2.],requires_grad=True)
        labels=torch.tensor([1.,0.])
        loss=pairwise_rank_loss(logits,labels)
        loss.backward()
        self.assertLess(logits.grad[0],0)
        self.assertGreater(logits.grad[1],0)
        self.assertGreater(loss.item(),pairwise_rank_loss(-logits,labels).item())
        torch.testing.assert_close(loss,pairwise_rank_loss(logits+10,labels))
        empty=pairwise_rank_loss(logits,torch.ones(2))
        self.assertEqual(empty.item(),0)
        self.assertTrue(empty.requires_grad)
        with self.assertRaises(ValueError):pairwise_rank_loss(logits,torch.tensor([.5,0.]))

    def test_configs_isolate_ranking_and_validate_incompatible_augmentation(self):
        a=Config.load('configs/experiments/shimmer_ranking_sampled_seed42.json').to_dict()
        b=Config.load('configs/experiments/shimmer_ranking_rank_seed42.json').to_dict()
        a.pop('experiment');b.pop('experiment')
        self.assertEqual(b['train'].pop('pairwise_rank_weight'),.1)
        a['train'].pop('pairwise_rank_weight')
        self.assertEqual(a,b)
        for changes in ({'pairwise_rank_weight':-1},{'pairwise_rank_weight':float('nan')},
                        {'pairwise_rank_weight':.1,'mixup_alpha':.2}):
            config=Config().to_dict();config['train'].update(changes)
            with self.assertRaises(ValueError):Config.from_dict(config)

    def test_resume_and_evaluation_provenance(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);fixture(root/'data')
            c=Config(device='cpu',output_root=str(root/'out'),model=ModelConfig(name='_test_tiny',params={}))
            c.data.root=str(root/'data');c.data.image_size=32;c.data.num_workers=0
            c.train.epochs=2;c.train.warmup_epochs=0;c.train.batch_size=3
            c.train.accumulation_steps=2;c.train.amp=False
            c.train.pairwise_rank_weight=.1;c.train.positive_fraction=.25
            full=train(c);partial=train(c,stop_after_epochs=1);resumed=train(c,resume=partial/'last.pt')
            a=read_checkpoint(full/'last.pt','training');b=read_checkpoint(resumed/'last.pt','training')
            history=json.loads((full/'history.json').read_text())
            self.assertGreater(sum(row['train']['rank_batches_with_pairs'] for row in history),0)
            self.assertTrue(all(row['train']['rank_loss'] >= 0 for row in history))
            for key,value in a['model_state'].items():
                torch.testing.assert_close(value,b['model_state'][key],rtol=0,atol=0)
            main(['evaluate','--checkpoint',str(full/'best.pt'),'--data-root',str(root/'data'),
                  '--device','cpu','--num-workers','0','--precision','fp32','--hash-images',
                  '--output-dir',str(root/'eval')])
            report=json.loads((root/'eval/provenance.json').read_text())
            self.assertEqual(len(report['ordered_source_images_sha256']),64)
            self.assertEqual(len(report['checkpoint_sha256']),64)
            self.assertFalse(report['amp_effective'])
            script=load_script('run_resolution_ablation')
            source=Config.load('configs/experiments/shimmer_512_pretrained_reference.json')
            configs=script.resolve_configs(source,root/'data',root/'r',('sampled','rank'),
                                            workers=0,recipe_family='ranking')
            self.assertEqual(configs['rank'].data.image_size,512)
