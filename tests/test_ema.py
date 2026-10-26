"""EMA must survive full resume and preserve ordinary inference checkpoints."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from test_baseline import fixture
from glaucoma.config import Config, TrainConfig
from glaucoma.engine.checkpoint import load_model, read_checkpoint
from glaucoma.engine.optimization import build_optimizer, train_one_epoch
from glaucoma.engine.trainer import train


class SmallClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 1),
            torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(1))
        self.head = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(4, 1))

    def forward(self, images, ages):
        return self.head(self.encoder(images)).squeeze(1)


class EMATests(unittest.TestCase):
    def test_ema_does_not_advance_on_a_skipped_amp_update(self):
        class SkipFirstUpdate:
            value, calls = 8.0, 0

            def scale(self, loss):
                return loss

            def unscale_(self, optimizer):
                pass

            def get_scale(self):
                return self.value

            def step(self, optimizer):
                self.calls += 1
                if self.calls > 1:
                    optimizer.step()

            def update(self):
                if self.calls == 1:
                    self.value /= 2

        model = SmallClassifier()
        ema = torch.optim.swa_utils.AveragedModel(model)
        settings = TrainConfig(amp=False, accumulation_steps=1)
        optimizer = build_optimizer(model, settings)
        rows = [{'image': torch.randn(3, 32, 32), 'age': torch.zeros(3),
                 'label': torch.tensor(float(index))} for index in range(2)]
        stats, step = train_one_epoch(model, torch.utils.data.DataLoader(rows, batch_size=1),
            optimizer, SkipFirstUpdate(), torch.device('cpu'), settings, 0, 2, 0, ema)
        self.assertEqual(step, 1)
        self.assertEqual(stats['skipped_optimizer_steps'], 1)
        self.assertEqual(int(ema.n_averaged), 1)

    def test_defaults_validation_and_backbone_group_rates(self):
        config = Config()
        self.assertIsNone(config.train.ema_decay)
        self.assertEqual(config.train.backbone_lr_scale, 1.0)
        for values in ({"ema_decay": 0}, {"ema_decay": 1}, {"ema_decay": float('nan')},
                       {"backbone_lr_scale": 0}, {"backbone_lr_scale": 1.1}):
            with self.assertRaises(ValueError):
                Config.from_dict({"train": values})
        model = SmallClassifier()
        settings = TrainConfig(learning_rate=1e-4, image_lr_scale=0.5, backbone_lr_scale=0.2)
        optimizer = build_optimizer(model, settings)
        rates = {id(p): group['lr'] for group in optimizer.param_groups for p in group['params']}
        for name, parameter in model.named_parameters():
            self.assertAlmostEqual(rates[id(parameter)], 1e-5 if name.startswith('encoder.') else 5e-5)

    def test_ema_and_optimizer_resume_match_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root / 'data')
            config = Config.from_dict({'device': 'cpu', 'output_root': str(root / 'out'),
                'data': {'root': str(root / 'data'), 'image_size': 32, 'num_workers': 0},
                'train': {'epochs': 3, 'warmup_epochs': 1, 'batch_size': 3,
                          'accumulation_steps': 2, 'amp': False, 'ema_decay': 0.8,
                          'backbone_lr_scale': 0.2}})
            with patch('glaucoma.engine.trainer.build_model', side_effect=lambda *args, **kwargs: SmallClassifier()):
                full = train(config)
                partial = train(config, stop_after_epochs=1)
                resumed = train(config, resume=partial / 'last.pt')
                a, b = [read_checkpoint(run / 'last.pt', 'training') for run in (full, resumed)]
                self.assertEqual(a['global_step'], b['global_step'])
                self.assertEqual(int(a['ema']['n_averaged']), a['global_step'])
                for field in ('model_state', 'ema'):
                    for key, value in a[field].items():
                        torch.testing.assert_close(value, b[field][key], atol=0, rtol=0)
                for left, right in zip(a['history'], b['history']):
                    for field in ('valid', 'valid_raw', 'valid_ema', 'selected_weights'):
                        self.assertEqual(left[field], right[field])
                best = read_checkpoint(resumed / 'best.pt', 'inference')
                self.assertIn(best['weights_variant'], ('raw', 'ema'))
                with patch('glaucoma.engine.checkpoint.build_model', side_effect=lambda *args, **kwargs: SmallClassifier()):
                    loaded, _, _ = load_model(resumed / 'best.pt', torch.device('cpu'))
                for key, value in loaded.state_dict().items():
                    torch.testing.assert_close(value, best['model_state'][key], atol=0, rtol=0)
                missing = copy.deepcopy(read_checkpoint(partial / 'last.pt', 'training'))
                del missing['ema']
                torch.save(missing, partial / 'missing-ema.pt')
                with self.assertRaisesRegex(ValueError, 'saved EMA state'):
                    train(config, resume=partial / 'missing-ema.pt')
