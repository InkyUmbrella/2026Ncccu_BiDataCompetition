"""Offline checks for pretrained initialization, preprocessing and checkpoint use."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import torch
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

from test_baseline import fixture
from glaucoma.cli import main
from glaucoma.config import Config
from glaucoma.data.age import AgeStats
from glaucoma.data.cache import ImageCache, prepare_cache
from glaucoma.data.dataset import read_samples
from glaucoma.data.transforms import build_transform
from glaucoma.engine.checkpoint import inference_payload, initialize_model, load_model, read_checkpoint
from glaucoma.engine.trainer import train
from glaucoma.models import build_model


class PretrainedTests(unittest.TestCase):
    def test_imagenet_b0_transfers_to_fwq_residual_without_downloading_weights(self):
        source_config = Config.load("configs/experiments/b0_crop384_imagenet.json")
        target_config = Config.load("configs/experiments/b0_crop384_age_residual_imagenet.json")
        legacy = Config.load("configs/experiments/b0_crop384_age_residual.json")
        self.assertEqual(target_config.train, legacy.train)
        self.assertEqual(target_config.model.params, legacy.model.params)
        self.assertEqual(target_config.data.normalization, source_config.data.normalization)
        for name in ("image_size", "crop_black_border", "crop_threshold", "crop_margin"):
            self.assertEqual(getattr(target_config.data, name), getattr(source_config.data, name))
        torch.manual_seed(42)
        with patch.object(ConvNeXt_Tiny_Weights, "get_state_dict", side_effect=AssertionError("network used")):
            source = build_model(source_config.model, load_pretrained=False).eval()
            payload = inference_payload(source, source_config, AgeStats(50, 50, 10), 4, {})
            target = build_model(target_config.model, load_pretrained=False).eval()
            initialize_model(target, target_config, payload)
        for key, value in source.state_dict().items():
            torch.testing.assert_close(target.state_dict()[key], value, atol=0, rtol=0)
        images, ages = torch.randn(2, 3, 32, 32), torch.randn(2, 3)
        with torch.inference_mode():
            torch.testing.assert_close(target(images, ages), source(images, ages), atol=0, rtol=0)
        wrong = copy.deepcopy(target_config)
        wrong.model.weights = None
        wrong.data.normalization = "symmetric"
        with self.assertRaisesRegex(ValueError, "matching normalization"):
            initialize_model(target, wrong, payload)

    def test_configs_preserve_old_defaults_and_isolate_initialization(self):
        legacy = Config.load("configs/baseline_b0.json")
        self.assertIsNone(legacy.model.weights)
        self.assertEqual(legacy.data.normalization, "symmetric")
        pretrained = Config.load("configs/experiments/b0_crop384_imagenet.json")
        control = Config.load("configs/experiments/b0_crop384_random_control.json")
        self.assertEqual(pretrained.model.weights, "IMAGENET1K_V1")
        values = pretrained.to_dict()
        values["experiment"] = control.experiment
        values["model"]["weights"] = None
        self.assertEqual(values, control.to_dict())
        invalid = pretrained.to_dict()
        invalid["data"]["normalization"] = "symmetric"
        with self.assertRaisesRegex(ValueError, "require imagenet"):
            Config.from_dict(invalid)
        invalid = pretrained.to_dict()
        invalid["model"]["weights"] = "DEFAULT"
        with self.assertRaisesRegex(ValueError, "weights must be"):
            Config.from_dict(invalid)
        invalid = pretrained.to_dict()
        invalid["model"]["params"]["weights"] = None
        with self.assertRaisesRegex(ValueError, "model.params.weights"):
            Config.from_dict(invalid)

    def test_imagenet_normalization_matches_official_values_and_lossless_cache(self):
        official = ConvNeXt_Tiny_Weights.IMAGENET1K_V1.transforms()
        image = Image.new("RGB", (32, 32), (51, 102, 153))
        expected = (torch.tensor([0.2, 0.4, 0.6]) - torch.tensor(official.mean)) / torch.tensor(official.std)
        actual = build_transform(32, normalization="imagenet")(image)
        torch.testing.assert_close(actual, expected[:, None, None].expand_as(actual))
        legacy = build_transform(32)(image)
        torch.testing.assert_close(legacy, (torch.tensor([0.2, 0.4, 0.6]) * 2 - 1)[:, None, None].expand_as(legacy))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root / "data")
            prepare_cache(root / "data", root / "cache", 32, 1, crop_black_border=True)
            rows = read_samples(root / "data", "train")
            cache = ImageCache(root / "cache", rows, 32, crop_black_border=True)
            for training in (False, True):
                with Image.open(rows[0].path) as original:
                    torch.manual_seed(42)
                    raw = build_transform(32, training, crop_black_border=True, normalization="imagenet")(original)
                torch.manual_seed(42)
                cached = build_transform(32, training, preprocessed=True, normalization="imagenet")(cache.load(rows[0].path))
                torch.testing.assert_close(raw, cached, rtol=0, atol=0)

    def test_real_encoder_receives_weights_and_offline_workflow_preserves_predictions(self):
        config = Config.load("configs/experiments/b0_crop384_imagenet.json")
        # Stand in for the official download while exercising torchvision's real loader.
        state = convnext_tiny(weights=None).state_dict()
        state["features.0.0.weight"].fill_(0.125)
        with patch.object(ConvNeXt_Tiny_Weights, "get_state_dict", return_value=state) as download:
            model = build_model(config.model)
        download.assert_called_once()
        torch.testing.assert_close(model.encoder[0][0][0].weight, state["features.0.0.weight"], rtol=0, atol=0)
        self.assertEqual(model.head[-1].out_features, 1)
        config.device = "cpu"
        config.data.image_size = 32
        config.data.num_workers = 0
        config.data.verify_images = False
        config.train.epochs = 2
        config.train.warmup_epochs = 0
        config.train.batch_size = 3
        config.train.accumulation_steps = 1
        config.train.amp = False
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root / "data")
            config.data.root = str(root / "data")
            config.output_root = str(root / "outputs")
            with patch.object(ConvNeXt_Tiny_Weights, "get_state_dict", return_value=state):
                partial = train(config, stop_after_epochs=1)
            with patch.object(ConvNeXt_Tiny_Weights, "get_state_dict", side_effect=AssertionError("network used")):
                resumed = train(config, resume=partial / "last.pt")
                initialized = train(config, init_from=partial / "best.pt", stop_after_epochs=1)
                loaded, saved, _ = load_model(resumed / "best.pt", torch.device("cpu"))
                self.assertEqual(saved.data.normalization, "imagenet")
                self.assertEqual(saved.model.weights, "IMAGENET1K_V1")
                self.assertEqual(read_checkpoint(initialized / "last.pt", "training")["epoch"], 0)
                main(["evaluate", "--checkpoint", str(resumed / "best.pt"), "--data-root", str(root / "data"),
                      "--device", "cpu", "--batch-size", "3", "--num-workers", "0", "--output-dir", str(root / "eval")])
                expected = json.loads((resumed / "metrics.json").read_text())["valid"]
                self.assertEqual(json.loads((root / "eval/metrics.json").read_text()), expected)
                main(["predict", "--checkpoint", str(resumed / "best.pt"), "--data-root", str(root / "data"),
                      "--device", "cpu", "--num-workers", "0", "--output", str(root / "predictions.csv")])
                payload = read_checkpoint(resumed / "best.pt", "inference")
                wrong = copy.deepcopy(config)
                wrong.model.weights = None
                wrong.data.normalization = "symmetric"
                with self.assertRaisesRegex(ValueError, "matching normalization"):
                    initialize_model(loaded, wrong, payload)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
