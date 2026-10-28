"""Portable recipe pairing, preparation, and independent evaluation contracts."""
import argparse
import csv
import json
from pathlib import Path
import tempfile
import unittest

from test_baseline import fixture
from test_fusion import load_script
from glaucoma.config import Config
from glaucoma.data.age import AgeStats
from glaucoma.engine.checkpoint import inference_payload, save_checkpoint
from glaucoma.models import build_model


class ResolutionTests(unittest.TestCase):
    def test_only_resolution_and_name_differ(self):
        script=load_script('run_resolution_ablation')
        source=Config.load('configs/experiments/shimmer_512_pretrained_reference.json')
        configs=script.resolve_configs(source,'/tmp/data','/tmp/results',(512,640))
        a,b=[configs[s].to_dict() for s in (512,640)]
        for c in (a,b):
            self.assertEqual(c['train']['batch_size']*c['train']['accumulation_steps'],128)
            self.assertEqual(c['train']['backbone_lr_scale'],.1)
            self.assertTrue(c['train']['keep_initial_best'])
            c.pop('experiment');c.pop('output_root');c['data'].pop('image_size')
        self.assertEqual(a,b)
        with self.assertRaises(ValueError):
            script.resolve_configs(source,'/tmp/data','/tmp/results',(512,),batch_size=7)
        source.data.decode='full'
        with self.assertRaisesRegex(ValueError,'preprocessing'):
            script.resolve_configs(source,'/tmp/data','/tmp/results',(640,))

    def test_prepare_without_gpu_and_evaluation_alignment(self):
        script=load_script('run_resolution_ablation')
        source=Config.load('configs/experiments/shimmer_512_pretrained_reference.json')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);fixture(root/'data')
            checkpoint=root/'best.pt'
            save_checkpoint(checkpoint,inference_payload(build_model(source.model,load_pretrained=False),
                            source,AgeStats(50,50,10),13,{}))
            args=argparse.Namespace(checkpoint=str(checkpoint),data_root=str(root/'data'),
                output_root=str(root/'result'),sizes='both',batch_size=4,workers=0,
                device='cpu',cache=False,prepare_only=True)
            script.main(args)
            manifest=json.loads((root/'result/manifest.json').read_text())
            self.assertEqual(manifest['status'],'prepared')
            self.assertEqual(len(manifest['checkpoint_sha256']),64)
            self.assertEqual(Config.load(root/'result/640/config.json').train.accumulation_steps,32)
            self.assertFalse((root/'result/640/training').exists())
            with self.assertRaises(FileExistsError):script.main(args)
            evaluation=root/'eval';evaluation.mkdir()
            (evaluation/'metrics.json').write_text('{"auroc":1.0}')
            (evaluation/'predictions.csv').write_text('Image_name,True_Label\na.png,1\nb.png,0\n')
            self.assertEqual(script.verify_evaluation(evaluation,[('a.png',1),('b.png',0)])['auroc'],1.)
            with self.assertRaises(ValueError):
                script.verify_evaluation(evaluation,[('b.png',0),('a.png',1)])
