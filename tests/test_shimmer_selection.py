"""Exercise raw/EMA winner selection without downloading weights or real images."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from test_baseline import fixture
from glaucoma.config import Config, ModelConfig
from glaucoma.engine.checkpoint import read_checkpoint
from glaucoma.engine.evaluation import collect_predictions
from glaucoma.engine.optimization import build_optimizer
from glaucoma.engine.trainer import train
from glaucoma.models import build_model


class SelectionTests(unittest.TestCase):
    def test_raw_and_ema_winners_export_the_evaluated_weights(self):
        for scores, winner in (((0.9, 0.8), 'raw'), ((0.8, 0.9), 'ema')):
            with self.subTest(winner=winner), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                fixture(root / 'data')
                config = Config(device='cpu', output_root=str(root / 'runs'),
                                model=ModelConfig(name='_test_tiny', params={}))
                config.data.root = str(root / 'data')
                config.data.image_size = 32
                config.data.num_workers = 0
                config.train.epochs = 1
                config.train.warmup_epochs = 0
                config.train.ema_decay = 0.999
                config.train.evaluate_raw_with_ema = True
                captured = []
                def evaluate(model, *args, **kwargs):
                    predictions, metrics = collect_predictions(model, *args, **kwargs)
                    if len(captured) < 2:
                        metrics['auroc'] = scores[len(captured)]
                    captured.append({k: v.detach().cpu().clone() for k,v in model.state_dict().items()})
                    return predictions, metrics
                with patch('glaucoma.engine.trainer.collect_predictions', side_effect=evaluate):
                    run = train(config)
                best = read_checkpoint(run / 'best.pt', 'inference')
                self.assertEqual(best['selected_weights'], winner)
                expected = captured[0 if winner == 'raw' else 1]
                for key,value in expected.items():
                    torch.testing.assert_close(best['model_state'][key],value,rtol=0,atol=0)

    def test_backbone_lr_scale_leaves_head_at_full_rate(self):
        config = Config(model=ModelConfig(name='_test_tiny', params={}))
        config.train.backbone_lr_scale = 0.2
        model = build_model(config.model)
        optimizer = build_optimizer(model, config.train)
        scales={id(p):g['lr_scale'] for g in optimizer.param_groups for p in g['params']}
        for name,p in model.named_parameters():
            self.assertEqual(scales[id(p)],0.2 if name.startswith('encoder.') else 1.0)
