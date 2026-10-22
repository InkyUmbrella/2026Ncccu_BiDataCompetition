"""Regression checks for residual-age transfer and matching crop preprocessing."""

import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from test_baseline import fixture
from glaucoma.cli import main
from glaucoma.config import Config, ModelConfig, TrainConfig
from glaucoma.data.age import AgeStats
from glaucoma.data.cache import ImageCache, prepare_cache
from glaucoma.data.dataset import FundusDataset, read_samples
from glaucoma.data.transforms import CropBlackBorder, build_transform
from glaucoma.engine.checkpoint import inference_payload, initialize_model, load_model, read_checkpoint, save_checkpoint
from glaucoma.engine.evaluation import collect_predictions
from glaucoma.engine.optimization import build_optimizer, train_one_epoch
from glaucoma.engine.trainer import make_loader, train
from glaucoma.models import build_model
from glaucoma.models.convnext import ConvNeXtAgeResidual


class PreprocessingTests(unittest.TestCase):
    def test_crop_preserves_field_margin_and_handles_dark_images(self):
        pixels = np.zeros((100, 120, 3), dtype=np.uint8)
        pixels[20:80, 30:90] = 100
        cropped = CropBlackBorder(10, 0.1)(Image.fromarray(pixels))
        self.assertEqual(cropped.size, (72, 72))
        self.assertEqual(int((np.asarray(cropped).max(axis=2) > 10).sum()), 3600)
        black = Image.new("RGB", (120, 100))
        self.assertEqual(CropBlackBorder()(black).size, black.size)
        black.putpixel((60, 50), (255, 255, 255))
        self.assertEqual(CropBlackBorder()(black).size, black.size)

    def test_cropped_cache_matches_original_once_and_rejects_wrong_recipe(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root / "data")
            for split in ("train", "valid"):
                for path in (root / "data" / split).iterdir():
                    with Image.open(path) as image:
                        canvas = Image.new("RGB", (150, 120))
                        canvas.paste(image, (50, 40))
                    canvas.save(path)
            prepare_cache(root / "data", root / "cache", 32, 1, crop_black_border=True)
            rows = read_samples(root / "data", "train")
            cache = ImageCache(root / "cache", rows, 32, crop_black_border=True)
            for training in (False, True):
                raw = FundusDataset(rows, AgeStats(50, 50, 10), build_transform(32, training, crop_black_border=True))
                cached = FundusDataset(rows, AgeStats(50, 50, 10), build_transform(32, training, preprocessed=True), cache)
                for index in (0, 2):
                    torch.manual_seed(index)
                    expected = raw[index]["image"]
                    torch.manual_seed(index)
                    torch.testing.assert_close(cached[index]["image"], expected, atol=0, rtol=0)
            with self.assertRaisesRegex(ValueError, "preprocessing mismatch"):
                ImageCache(root / "cache", rows, 32)
            with self.assertRaisesRegex(ValueError, "preprocessing mismatch"):
                ImageCache(root / "cache", rows, 32, crop_black_border=True, crop_margin=0.05)
            with self.assertRaisesRegex(ValueError, "image size mismatch"):
                ImageCache(root / "cache", rows, 64, crop_black_border=True)

    def test_legacy_cache_only_supports_original_preprocessing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root / "data")
            prepare_cache(root / "data", root / "cache", 32, 1)
            manifest = root / "cache" / "manifest.json"
            value = json.loads(manifest.read_text())
            value["version"] = 1
            value.pop("preprocessing")
            manifest.write_text(json.dumps(value))
            rows = read_samples(root / "data", "train")
            ImageCache(root / "cache", rows, 32)
            with self.assertRaisesRegex(ValueError, "preprocessing mismatch"):
                ImageCache(root / "cache", rows, 32, crop_black_border=True)


class TransferTests(unittest.TestCase):
    def test_config_and_initialization_guards(self):
        config = Config.load("configs/experiments/b0_crop384_age_residual.json")
        config.device = "cpu"
        self.assertEqual(config.train.freeze_image_epochs, 3)
        self.assertEqual(config.train.image_lr_scale, 0.1)
        self.assertTrue(config.data.crop_black_border)
        for updates in ({"freeze_image_epochs": -1}, {"freeze_image_epochs": 20},
                        {"image_lr_scale": 0}, {"image_lr_scale": float("nan")}):
            values = config.to_dict()
            values["train"].update(updates)
            with self.assertRaises(ValueError):
                Config.from_dict(values)
        with self.assertRaisesRegex(ValueError, "requires init_from, init_encoder or resume"):
            train(config)
        with self.assertRaisesRegex(ValueError, "Expected version 1 inference"):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "last.pt"
                save_checkpoint(path, {"format_version": 1, "kind": "training"})
                train(config, init_from=path)

    def test_legacy_optimizer_groups_load_with_zero_weight_decay(self):
        # Bias first deliberately differs from decay-group ordering.
        model = torch.nn.Module()
        model.bias = torch.nn.Parameter(torch.zeros(2))
        model.weight = torch.nn.Parameter(torch.ones(2, 2))
        settings = TrainConfig(weight_decay=0.0)
        legacy = torch.optim.AdamW([{"params": [model.weight], "weight_decay": 0.0},
                                    {"params": [model.bias], "weight_decay": 0.0}],
                                   lr=settings.learning_rate)
        (model.bias.sum() + model.weight.sum()).backward()
        legacy.step()
        current = build_optimizer(model, settings)
        current.load_state_dict(legacy.state_dict())
        self.assertEqual(len(current.param_groups), 2)
        self.assertIs(current.param_groups[0]["params"][0], model.weight)
        self.assertIs(current.param_groups[1]["params"][0], model.bias)
        torch.testing.assert_close(current.state[model.weight]["exp_avg"],
                                   legacy.state[model.weight]["exp_avg"])

    def test_b0_transfer_zero_correction_freeze_and_unfreeze(self):
        torch.manual_seed(42)
        b0_config = Config(model=ModelConfig(params={"use_age": False}))
        b0 = build_model(b0_config.model).eval()
        config = Config(model=ModelConfig(name="convnext_tiny_age_residual", params={}))
        residual = build_model(config.model)
        payload = inference_payload(b0, b0_config, AgeStats(50, 50, 10), 4, {})
        initialize_model(residual, config, payload)
        images, ages = torch.randn(2, 3, 32, 32), torch.randn(2, 3)
        residual.eval()
        with torch.no_grad():
            torch.testing.assert_close(b0(images, ages), residual(images, ages), atol=0, rtol=0)
            torch.testing.assert_close(residual(images, ages), residual(images, ages + 4), atol=0, rtol=0)
        settings = TrainConfig(amp=False, learning_rate=0.01, image_lr_scale=0.1)
        optimizer = build_optimizer(residual, settings)
        residual.set_image_frozen(True)
        before = copy.deepcopy(residual.encoder.state_dict())
        head_before = copy.deepcopy(residual.head.state_dict())
        # Exercise the training loop with a real DataLoader (including sample normalization).
        dataset = [{"image": image, "age": age, "label": torch.tensor(float(i))}
                   for i, (image, age) in enumerate(zip(images, ages))]
        loader = torch.utils.data.DataLoader(dataset, batch_size=2)
        train_one_epoch(residual, loader, optimizer, torch.amp.GradScaler("cuda", enabled=False),
                        torch.device("cpu"), settings, 0, 2, 0)
        self.assertFalse(residual.head.training)
        for key, value in before.items():
            torch.testing.assert_close(residual.encoder.state_dict()[key], value, atol=0, rtol=0)
        for key, value in head_before.items():
            torch.testing.assert_close(residual.head.state_dict()[key], value, atol=0, rtol=0)
        self.assertGreater(residual.age_residual[-1].weight.abs().sum().item(), 0)
        image_groups = [group for group in optimizer.param_groups if group["lr_scale"] == 0.1]
        self.assertTrue(image_groups)
        self.assertAlmostEqual(image_groups[0]["lr"], settings.learning_rate * 0.1)
        residual.set_image_frozen(False)
        residual.train()
        self.assertTrue(residual.head.training)
        self.assertTrue(all(parameter.requires_grad for parameter in residual.parameters()))
        # A trained residual must survive checkpoint loading and the public CLI.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root / "data")
            config.data.image_size = 32
            config.data.crop_black_border = True
            config.train.amp = False
            checkpoint = root / "best.pt"
            save_checkpoint(checkpoint, inference_payload(residual, config, AgeStats(50, 50, 10), 0, {}))
            loaded, loaded_config, stats = load_model(checkpoint, torch.device("cpu"))
            self.assertTrue(loaded_config.data.crop_black_border)
            self.assertEqual(stats, AgeStats(50, 50, 10))
            residual.eval()
            with torch.no_grad():
                torch.testing.assert_close(loaded(images, ages), residual(images, ages), atol=0, rtol=0)
            rows = read_samples(root / "data", "valid")
            dataset = FundusDataset(rows, stats, build_transform(32, crop_black_border=True))
            _, expected = collect_predictions(loaded, make_loader(dataset, 2, 0, 42),
                                              torch.device("cpu"), config.train.pos_weight, False)
            main(["evaluate", "--checkpoint", str(checkpoint), "--data-root", str(root / "data"),
                  "--device", "cpu", "--batch-size", "2", "--num-workers", "0",
                  "--output-dir", str(root / "evaluation")])
            self.assertEqual(json.loads((root / "evaluation/metrics.json").read_text()), expected)
            main(["predict", "--checkpoint", str(checkpoint), "--data-root", str(root / "data"),
                  "--device", "cpu", "--num-workers", "0", "--output", str(root / "submission.csv")])
            with (root / "submission.csv").open() as handle:
                predictions = list(csv.DictReader(handle))
            self.assertEqual([row["Image_name"] for row in predictions],
                             [row.name for row in read_samples(root / "data", "test", False)])
            self.assertTrue(all(0 <= float(row["Probability"]) <= 1 for row in predictions))
        payload["config"]["model"]["params"]["use_age"] = True
        with self.assertRaisesRegex(ValueError, "Initialization requires"):
            initialize_model(residual, config, payload)

    def test_new_experiment_resets_state_and_sampling_resume_is_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root / "data")
            config = Config.from_dict({"device": "cpu", "output_root": str(root / "out"),
                "data": {"root": str(root / "data"), "image_size": 32, "num_workers": 0},
                "model": {"name": "_test_tiny", "params": {}},
                "train": {"epochs": 3, "warmup_epochs": 0, "batch_size": 2,
                          "amp": False, "positive_fraction": 0.125, "pos_weight": 1.0}})
            full = train(config)
            partial = train(config, stop_after_epochs=1)
            resumed = train(config, resume=partial / "last.pt")
            a, b = [read_checkpoint(path / "last.pt", "training") for path in (full, resumed)]
            for key, value in a["model_state"].items():
                torch.testing.assert_close(value, b["model_state"][key], atol=0, rtol=0)
            changed = copy.deepcopy(config)
            changed.train.epochs = 4
            changed.data.crop_black_border = True
            initialized = train(changed, init_from=full / "best.pt", stop_after_epochs=1)
            fresh = read_checkpoint(initialized / "last.pt", "training")
            self.assertEqual(fresh["epoch"], 0)
            self.assertEqual(len(fresh["history"]), 1)
            self.assertEqual(fresh["global_step"], 2)
            self.assertEqual(fresh["age_stats"], a["age_stats"])
            with self.assertRaisesRegex(ValueError, "mutually exclusive"):
                train(config, resume=partial / "last.pt", init_from=full / "best.pt")
            with self.assertRaisesRegex(ValueError, "config differs"):
                train(changed, resume=partial / "last.pt")

    def test_residual_resume_crosses_frozen_boundary_without_changing_weights(self):
        class SmallResidual(ConvNeXtAgeResidual):
            def __init__(self):
                torch.nn.Module.__init__(self)
                self.use_age = False
                self.encoder = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 1),
                    torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(1))
                self.head = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(4, 1))
                self.age_residual = torch.nn.Sequential(torch.nn.Linear(3, 16),
                    torch.nn.GELU(), torch.nn.Linear(16, 1))
                torch.nn.init.zeros_(self.age_residual[-1].weight)
                torch.nn.init.zeros_(self.age_residual[-1].bias)
                self.image_frozen = False

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root / "data")
            config = Config.from_dict({"device": "cpu", "output_root": str(root / "out"),
                "data": {"root": str(root / "data"), "image_size": 32, "num_workers": 0},
                "model": {"name": "convnext_tiny_age_residual", "params": {}},
                "train": {"epochs": 4, "warmup_epochs": 1, "batch_size": 3,
                          "accumulation_steps": 1, "amp": False, "freeze_image_epochs": 2,
                          "image_lr_scale": 0.1}})
            initial = SmallResidual()
            checkpoint = root / "initial.pt"
            save_checkpoint(checkpoint, inference_payload(initial, config, AgeStats(50, 50, 10), 0, {}))
            with patch("glaucoma.engine.trainer.build_model", side_effect=lambda _: SmallResidual()):
                full = train(config, init_from=checkpoint)
                partial = train(config, init_from=checkpoint, stop_after_epochs=1)
                resumed = train(config, resume=partial / "last.pt")
            a, b, frozen = [read_checkpoint(path / "last.pt", "training")
                            for path in (full, resumed, partial)]
            self.assertEqual([row["image_frozen"] for row in a["history"]], [True, True, False, False])
            for key, value in a["model_state"].items():
                torch.testing.assert_close(value, b["model_state"][key], atol=0, rtol=0)
            for key, value in initial.state_dict().items():
                if key.startswith(("encoder.", "head.")):
                    torch.testing.assert_close(value, frozen["model_state"][key], atol=0, rtol=0)
            self.assertTrue(any(not torch.equal(value, a["model_state"][key])
                                for key, value in initial.state_dict().items() if key.startswith("encoder.")))


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
