"""Real EfficientNet transfer/lifecycle and refinement fallback, using synthetic data only."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import torch
from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s

from test_baseline import fixture
from test_fusion import load_script
from glaucoma.cli import main
from glaucoma.config import Config, ModelConfig
from glaucoma.data.age import AgeStats
from glaucoma.data.transforms import build_transform
from glaucoma.engine.checkpoint import inference_payload, read_checkpoint, save_checkpoint, load_model
from glaucoma.engine.evaluation import collect_predictions
from glaucoma.engine.trainer import train
from glaucoma.models import build_model


class EfficientNetTests(unittest.TestCase):
    def test_official_normalization_transfer_and_offline_lifecycle(self):
        config = Config.load('configs/experiments/shimmer_efficientnetv2s_384_reference_seed42.json')
        official = EfficientNet_V2_S_Weights.IMAGENET1K_V1.transforms()
        transformed = build_transform(32, normalization=config.data.normalization)(Image.new('RGB', (32,32), (51,102,153)))
        expected = (torch.tensor([.2,.4,.6])-torch.tensor(official.mean))/torch.tensor(official.std)
        torch.testing.assert_close(transformed, expected[:,None,None].expand_as(transformed))
        state = efficientnet_v2_s(weights=None).state_dict()
        state['features.0.0.weight'].fill_(0.0125)
        with patch.object(EfficientNet_V2_S_Weights, 'get_state_dict', return_value=state) as download:
            model = build_model(config.model)
        download.assert_called_once()
        torch.testing.assert_close(model.encoder[0][0][0].weight, state['features.0.0.weight'], rtol=0, atol=0)
        self.assertEqual(model.head[-1].out_features, 1)
        config.device='cpu'
        config.data.image_size=32
        config.data.num_workers=0
        config.data.verify_images=False
        config.train.epochs=2
        config.train.warmup_epochs=0
        config.train.batch_size=3
        config.train.accumulation_steps=1
        config.train.amp=False
        config.train.ema_decay=.99
        config.train.evaluate_raw_with_ema=True
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);fixture(root/'data')
            config.data.root=str(root/'data');config.output_root=str(root/'runs')
            with patch.object(EfficientNet_V2_S_Weights, 'get_state_dict', return_value=state):
                partial=train(config,stop_after_epochs=1)
            with patch.object(EfficientNet_V2_S_Weights, 'get_state_dict', side_effect=AssertionError('network used')):
                resumed=train(config,resume=partial/'last.pt')
                initialized=train(config,init_from=partial/'best.pt',stop_after_epochs=1)
                loaded,saved,_=load_model(resumed/'best.pt',torch.device('cpu'))
                self.assertEqual(saved.model.name,'efficientnet_v2_s')
                self.assertEqual(loaded(torch.zeros(2,3,32,32),torch.zeros(2,3)).shape,(2,))
                main(['evaluate','--checkpoint',str(resumed/'best.pt'),'--data-root',str(root/'data'),
                      '--device','cpu','--num-workers','0','--batch-size','3','--output-dir',str(root/'eval')])
                expected=json.loads((resumed/'metrics.json').read_text())['valid']
                self.assertEqual(json.loads((root/'eval/metrics.json').read_text()),expected)
                main(['predict','--checkpoint',str(initialized/'best.pt'),'--data-root',str(root/'data'),
                      '--device','cpu','--num-workers','0','--output',str(root/'predictions.csv')])


class RefineTests(unittest.TestCase):
    def test_initial_weights_remain_best_after_worse_training_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);fixture(root/'data')
            config=Config(device='cpu',output_root=str(root/'runs'),model=ModelConfig(name='_test_tiny',params={}))
            config.data.root=str(root/'data');config.data.image_size=32;config.data.num_workers=0
            config.train.epochs=2;config.train.warmup_epochs=0
            config.train.keep_initial_best=True
            original=inference_payload(build_model(config.model),config,AgeStats(50,50,10),0,{})
            save_checkpoint(root/'source.pt',original)
            calls=0
            def evaluate(model,*args,**kwargs):
                nonlocal calls
                predictions,metrics=collect_predictions(model,*args,**kwargs)
                metrics['auroc']=.99 if calls==0 else .5
                calls+=1
                return predictions,metrics
            with patch('glaucoma.engine.trainer.collect_predictions',side_effect=evaluate):
                partial=train(config,init_from=root/'source.pt',stop_after_epochs=1)
                resumed=train(config,resume=partial/'last.pt')
            best=read_checkpoint(resumed/'best.pt','inference')
            self.assertEqual(best['epoch'],-1)
            self.assertEqual(best['selected_weights'],'initial')
            for key,value in original['model_state'].items():
                torch.testing.assert_close(best['model_state'][key],value,rtol=0,atol=0)
            self.assertTrue((partial/'initial_validation.json').exists())
            metadata=json.loads((partial/'metadata.json').read_text())
            self.assertEqual(len(metadata['initialized_from_sha256']),64)

    def test_refine_recipe_preserves_successful_model_and_preprocessing(self):
        baseline=Config.load('configs/experiments/shimmer_512_pretrained_reference.json')
        refine=Config.load('configs/experiments/shimmer_512_reference_refine_seed42.json')
        self.assertEqual(baseline.model,refine.model)
        self.assertEqual(baseline.data,refine.data)
        self.assertEqual(refine.train.positive_fraction,None)
        self.assertEqual(refine.train.mixup_alpha,None)
        self.assertEqual(refine.train.label_smoothing,0)
        self.assertTrue(refine.train.keep_initial_best)
        with self.assertRaisesRegex(ValueError,'requires init_from'):
            train(refine)


class ReportTests(unittest.TestCase):
    def test_flat_evaluation_derives_epoch_and_partial_completion_from_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'config.json').write_text(json.dumps({'experiment':'partial','train':{'epochs':30}}))
            (root/'history.json').write_text(json.dumps([
                {'epoch':0,'valid':{'auroc':.8,'auprc_trapz':.5}},
                {'epoch':1,'valid':{'auroc':.9,'auprc_trapz':.6}}]))
            (root/'metrics.json').write_text(json.dumps({'auroc':.9,'auprc_trapz':.6}))
            row=load_script('report_runs').scan_runs(root)[0]
            self.assertEqual(row['best_epoch_one_based'],2)
            self.assertEqual(row['completed_epochs'],2)
            self.assertEqual(row['planned_epochs'],30)
            self.assertFalse(row['training_complete'])
            (root/'metrics.json').write_text(json.dumps({'auroc':.91,'auprc_trapz':.6}))
            row=load_script('report_runs').scan_runs(root)[0]
            self.assertIsNone(row['best_epoch'])


class GPUGuardTests(unittest.TestCase):
    def test_only_allocated_gpus_are_queried_and_busy_cards_are_not_used(self):
        runner=load_script('run_round2_server')
        with patch.object(runner.subprocess,'check_output') as query:
            with self.assertRaisesRegex(ValueError,'restricted'):
                runner.free_gpu(0)
            query.assert_not_called()
        with patch.object(runner.subprocess,'check_output',side_effect=['1234\n','']), \
             patch.object(runner.time,'sleep'):
            runner.free_gpu(2)
        with patch.object(runner.subprocess,'check_output',return_value='1234\n'), \
             patch.object(runner.time,'monotonic',side_effect=[0,121]):
            with self.assertRaisesRegex(RuntimeError,'still occupied'):
                runner.free_gpu(3)
