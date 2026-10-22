"""Default-off trainer upgrades: EMA, stochastic depth, Mixup/CutMix, label smoothing.

Synthetic data only: no competition images, network or pretrained downloads.
"""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from test_baseline import fixture

from glaucoma.config import Config, ModelConfig, TrainConfig
from glaucoma.engine.checkpoint import read_checkpoint
from glaucoma.engine.ema import ModelEMA, apply_ema
from glaucoma.engine.optimization import mix_batch, smooth_labels, train_one_epoch
from glaucoma.engine.trainer import train
from glaucoma.models import build_model

ROOT = Path(__file__).resolve().parents[1]


class LinearClassifier(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.layer = nn.Linear(2, 1)

    def forward(self, image, age):
        return self.layer(self.dropout(image)).squeeze(1)


class Samples(Dataset):
    def __len__(self):
        return 6

    def __getitem__(self, index):
        return {"image": torch.tensor([index / 6, 0.5]), "age": torch.zeros(3),
                "label": torch.tensor(float(index % 2))}


def scaler(scale_factor=1.0):
    """Minimal GradScaler stand-in; scale_factor 1 keeps steps, 0.5 emulates a skip."""
    class StandIn:
        def __init__(self):
            self.scale_value = 1.0
            self.factor = scale_factor

        def scale(self, value):
            return value

        def unscale_(self, optimizer):
            pass

        def step(self, optimizer):
            if self.factor >= 1.0:
                optimizer.step()

        def update(self):
            self.scale_value *= self.factor

        def get_scale(self):
            return self.scale_value

    return StandIn()


class EmaTests(unittest.TestCase):
    def test_ema_update_warmup_and_shadow_round_trip(self):
        torch.manual_seed(0)
        model = nn.Linear(3, 2)
        ema = ModelEMA(model, 0.9)
        before = model.weight.detach().clone()
        with torch.no_grad():
            model.weight.add_(1.0)
        ema.update(model)
        torch.testing.assert_close(ema.shadow["weight"], 0.9 * before + 0.1 * model.weight.detach())
        self.assertEqual(ema.num_updates, 1)
        self.assertEqual(ema.shadow["weight"].dtype, torch.float32)

        plain = ModelEMA(model, 0.9)
        self.assertEqual(plain.current_decay(0), 0.9)
        warm = ModelEMA(model, 0.9, warmup_steps=5)
        self.assertAlmostEqual(warm.current_decay(0), 0.1)
        self.assertAlmostEqual(warm.current_decay(5), 6 / 15)
        self.assertEqual(warm.current_decay(100), 0.9)

        state = ema.state_dict()
        restored = ModelEMA(model, 0.9)
        restored.load_state_dict(state)
        self.assertEqual(restored.num_updates, ema.num_updates)
        for key, value in ema.shadow.items():
            torch.testing.assert_close(restored.shadow[key], value, rtol=0, atol=0)

        with torch.no_grad():
            model.weight.fill_(7.0)
        with apply_ema(ema, model):
            torch.testing.assert_close(model.weight, ema.shadow["weight"], rtol=0, atol=0)
        torch.testing.assert_close(model.weight, torch.full_like(model.weight, 7.0), rtol=0, atol=0)

    def test_ema_skips_amp_skipped_steps(self):
        torch.manual_seed(1)
        model = LinearClassifier()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        config = TrainConfig(accumulation_steps=1, learning_rate=0.1, min_learning_rate=0.1,
                             grad_clip=100, amp=False)
        loader = DataLoader(Samples(), batch_size=3)
        skipped_ema = ModelEMA(model, 0.9)
        _, step = train_one_epoch(model, loader, optimizer, scaler(scale_factor=0.5), torch.device("cpu"),
                                  config, 0, 2, 0, ema=skipped_ema)
        self.assertEqual(step, 0)
        self.assertEqual(skipped_ema.num_updates, 0)
        torch.testing.assert_close(skipped_ema.shadow["layer.weight"], model.layer.weight.detach())

        stepped_ema = ModelEMA(model, 0.9)
        _, step = train_one_epoch(model, loader, optimizer, scaler(scale_factor=1.0), torch.device("cpu"),
                                  config, 0, 4, 0, ema=stepped_ema)
        self.assertEqual(step, 2)
        self.assertEqual(stepped_ema.num_updates, 2)

    def test_ema_training_and_resume_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root / "data")
            config = Config.from_dict({
                "device": "cpu", "output_root": str(root / "outputs"),
                "data": {"root": str(root / "data"), "image_size": 32, "num_workers": 0},
                "model": {"name": "_test_tiny", "params": {}},
                "train": {"epochs": 2, "warmup_epochs": 0, "batch_size": 2,
                          "accumulation_steps": 2, "learning_rate": 0.01, "amp": False,
                          "ema_decay": 0.9, "ema_warmup_steps": 2,
                          "mixup_alpha": 0.2, "label_smoothing": 0.05}})
            full = train(config)
            partial = train(config, stop_after_epochs=1)
            resumed = train(config, resume=partial / "last.pt")
            a = read_checkpoint(full / "last.pt", "training")
            b = read_checkpoint(resumed / "last.pt", "training")
            self.assertEqual(a["global_step"], b["global_step"])
            self.assertEqual(a["ema"]["num_updates"], a["global_step"])
            self.assertEqual(a["ema"]["num_updates"], b["ema"]["num_updates"])
            for key, value in a["model_state"].items():
                torch.testing.assert_close(value, b["model_state"][key], rtol=0, atol=0)
            for key, value in a["ema"]["shadow"].items():
                torch.testing.assert_close(value, b["ema"]["shadow"][key], rtol=0, atol=0)
            self.assertTrue(any(not torch.equal(a["ema"]["shadow"][key], a["model_state"][key])
                                for key in a["ema"]["shadow"]))
            if a["best"]["epoch"] == a["epoch"]:
                for key, value in a["best"]["model_state"].items():
                    if value.is_floating_point():
                        torch.testing.assert_close(value, a["ema"]["shadow"][key], rtol=0, atol=0)


class MixupTests(unittest.TestCase):
    def test_mixup_and_cutmix_soft_targets_and_determinism(self):
        labels = torch.tensor([0., 1., 0., 1., 1., 0.])
        for seed in (3, 4):
            torch.manual_seed(seed)
            images = torch.randn(6, 3, 8, 8)
            torch.manual_seed(7)
            mixed, targets, info = mix_batch(images, labels, mixup_alpha=0.4)
            lam, permutation = info["lambda"], info["permutation"]
            self.assertEqual(info["transform"], "mixup")
            torch.testing.assert_close(targets, lam * labels + (1 - lam) * labels[permutation])
            torch.testing.assert_close(mixed, lam * images + (1 - lam) * images[permutation])
            self.assertTrue(((targets >= 0) & (targets <= 1)).all())

            torch.manual_seed(7)
            again, targets_again, _ = mix_batch(images, labels, mixup_alpha=0.4)
            torch.testing.assert_close(again, mixed, rtol=0, atol=0)
            torch.testing.assert_close(targets_again, targets, rtol=0, atol=0)

        torch.manual_seed(9)
        cut_images, cut_targets, cut_info = mix_batch(images, labels, cutmix_alpha=1.0)
        self.assertEqual(cut_info["transform"], "cutmix")
        self.assertFalse(torch.equal(cut_images, images))
        self.assertTrue(((cut_targets >= 0) & (cut_targets <= 1)).all())
        torch.manual_seed(9)
        _, _, both = mix_batch(images, labels, mixup_alpha=0.4, cutmix_alpha=1.0)
        self.assertIn(both["transform"], ("mixup", "cutmix"))

    def test_soft_target_pos_weight_semantics(self):
        logits = torch.tensor([-0.5, 1.3, 0.2])
        soft = torch.tensor([0.25, 0.75, 0.5])
        weight = 5.0
        loss = F.binary_cross_entropy_with_logits(logits, soft, pos_weight=torch.tensor(weight))
        probability = logits.sigmoid()
        expected = -(weight * soft * probability.log() + (1 - soft) * (1 - probability).log()).mean()
        torch.testing.assert_close(loss, expected)
        # The positive weight scales the soft positive mass, not the sample count.
        hard = F.binary_cross_entropy_with_logits(logits, torch.tensor([0., 1., 0.]),
                                                  pos_weight=torch.tensor(weight))
        self.assertFalse(torch.equal(loss, hard))

    def test_label_smoothing_helper_and_training_loss(self):
        labels = torch.tensor([0., 1., 0.])
        torch.testing.assert_close(smooth_labels(labels, 0.2), torch.tensor([0.1, 0.9, 0.1]))
        self.assertIs(smooth_labels(labels, 0.0), labels)

        torch.manual_seed(5)
        model = LinearClassifier()
        config = TrainConfig(accumulation_steps=1, learning_rate=0.1, min_learning_rate=0.1,
                             grad_clip=100, amp=False, label_smoothing=0.2)
        loader = DataLoader(Samples(), batch_size=3)
        model.train()
        with torch.no_grad():
            total, seen = 0.0, 0
            for batch in loader:
                logits = model(batch["image"], batch["age"])
                total += F.binary_cross_entropy_with_logits(
                    logits, smooth_labels(batch["label"], 0.2),
                    pos_weight=torch.tensor(config.pos_weight), reduction="sum").item()
                seen += len(batch["label"])
            expected = total / seen
        metrics, _ = train_one_epoch(model, loader, torch.optim.SGD(model.parameters(), lr=0.1),
                                     scaler(scale_factor=0.5), torch.device("cpu"), config, 0, 2, 0)
        self.assertAlmostEqual(metrics["loss"], expected, places=6)


class ModelUpgradeTests(unittest.TestCase):
    def test_stochastic_depth_plumbing(self):
        def build(**params):
            torch.manual_seed(42)
            return build_model(ModelConfig(params=params))

        default = build(use_age=False)
        explicit_zero = build(use_age=False, stochastic_depth_prob=0.0)
        deep = build(use_age=False, stochastic_depth_prob=0.3)
        for model in (default, explicit_zero):
            self.assertFalse(any(type(module).__name__ == "StochasticDepth" and module.p > 0
                                 for module in model.modules()))
        self.assertGreater(sum(type(module).__name__ == "StochasticDepth"
                               for module in default.modules()), 0)
        self.assertGreater(sum(type(module).__name__ == "StochasticDepth"
                               for module in deep.modules()), 0)
        self.assertTrue(any(type(module).__name__ == "StochasticDepth" and module.p > 0
                            for module in deep.modules()))
        default_state = default.state_dict()
        for model in (explicit_zero, deep):
            state = model.state_dict()
            self.assertEqual(set(state), set(default_state))
            for key, value in state.items():
                torch.testing.assert_close(value, default_state[key], rtol=0, atol=0)
        # In evaluation mode stochastic depth is the identity, so outputs are unchanged.
        images, ages = torch.randn(2, 3, 32, 32), torch.randn(2, 3)
        default.eval()
        deep.eval()
        with torch.no_grad():
            torch.testing.assert_close(default(images, ages), deep(images, ages), rtol=0, atol=0)

    def test_config_defaults_and_legacy_config_compatibility(self):
        train = TrainConfig()
        self.assertIsNone(train.ema_decay)
        self.assertEqual(train.ema_warmup_steps, 0)
        self.assertIsNone(train.mixup_alpha)
        self.assertIsNone(train.cutmix_alpha)
        self.assertEqual(train.label_smoothing, 0.0)

        legacy = json.loads((ROOT / "configs/experiments/b0_crop384_control.json").read_text())
        loaded = Config.from_dict(legacy)
        self.assertIsNone(loaded.train.ema_decay)
        self.assertEqual(loaded.train.label_smoothing, 0.0)
        self.assertEqual(Config.from_dict(json.loads(json.dumps(loaded.to_dict()))).to_dict(),
                         loaded.to_dict())

        for bad in ({"ema_decay": 0}, {"ema_decay": 1}, {"ema_warmup_steps": -1},
                    {"mixup_alpha": 0}, {"cutmix_alpha": -2}, {"label_smoothing": 0.6}):
            with self.assertRaises(ValueError):
                Config.from_dict({"train": bad})

    def test_defaults_preserve_original_math_and_rng(self):
        """Default-off config must follow the pre-upgrade loop and draw no extra RNG."""
        loader = DataLoader(Samples(), batch_size=2)
        config = TrainConfig(accumulation_steps=1, learning_rate=0.1, min_learning_rate=0.1,
                             grad_clip=100, amp=False)
        torch.manual_seed(11)
        upgraded = LinearClassifier(dropout=0.2)
        reference = copy.deepcopy(upgraded)
        loop_rng = torch.get_rng_state()
        optimizer = torch.optim.SGD(upgraded.parameters(), lr=0.1)
        train_one_epoch(upgraded, loader, optimizer, torch.amp.GradScaler("cuda", enabled=False),
                        torch.device("cpu"), config, 0, 3, 0)
        upgraded_rng = torch.get_rng_state()

        torch.set_rng_state(loop_rng)
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
        for batch in loader:
            reference_optimizer.zero_grad()
            logits = reference(batch["image"], batch["age"])
            loss = F.binary_cross_entropy_with_logits(logits, batch["label"],
                                                      pos_weight=torch.tensor(config.pos_weight),
                                                      reduction="sum") / len(batch["label"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(reference.parameters(), config.grad_clip)
            reference_optimizer.step()
        reference_rng = torch.get_rng_state()
        for a, b in zip(upgraded.parameters(), reference.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertTrue(torch.equal(upgraded_rng, reference_rng))


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
