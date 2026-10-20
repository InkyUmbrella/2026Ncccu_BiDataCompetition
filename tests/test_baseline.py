"""Synthetic tests only: no competition images, network or pretrained downloads."""

import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from glaucoma.cli import main
from glaucoma.config import Config, ModelConfig, TrainConfig
from glaucoma.data.age import AgeStats
from glaucoma.data.audit import audit_samples
from glaucoma.data.cache import ImageCache, prepare_cache
from glaucoma.data.dataset import FundusDataset, read_samples
from glaucoma.data.transforms import build_transform
from glaucoma.engine.checkpoint import load_model, read_checkpoint
from glaucoma.engine.optimization import learning_rate, train_one_epoch
from glaucoma.engine.trainer import make_loader, train
from glaucoma.engine.evaluation import collect_predictions
from glaucoma.metrics import binary_metrics
from glaucoma.models import build_model, register_model


@register_model("_test_tiny")
class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(3, 4, 1), nn.GELU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(1))
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(7, 1))

    def forward(self, image, age):
        return self.head(torch.cat((self.encoder(image), age), 1)).squeeze(1)


def fixture(root):
    for split, count in (("train", 9), ("valid", 4), ("test", 3)):
        directory = root / split
        directory.mkdir(parents=True)
        with (root / f"{split}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Image_name", "Age"] + ([] if split == "test" else ["True_Label"]))
            # Reverse order tests that inference follows the CSV, not directory order.
            for index in reversed(range(count)):
                name = f"{split}_{index:03d}"
                label = index % 2
                image = Image.new("RGB", (45 + index, 33), (30 + label * 180, 20 + index, 40))
                image.save(directory / f"{name}.JPG")
                writer.writerow([f"{name}.jpg", 40 + label * 30] + ([] if split == "test" else [label]))


class DataTests(unittest.TestCase):
    def test_lossless_cache_preserves_augmentation_and_rejects_stale_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root / "data")
            prepare_cache(root / "data", root / "cache", 32, 1)
            rows = read_samples(root / "data", "train")
            cache = ImageCache(root / "cache", rows, 32)
            for training in (False, True):
                original = FundusDataset(rows, AgeStats(50, 50, 10), build_transform(32, training))
                cached = FundusDataset(rows, AgeStats(50, 50, 10), build_transform(32, training), cache)
                for index in range(len(rows)):
                    torch.manual_seed(index)
                    expected = original[index]["image"]
                    torch.manual_seed(index)
                    torch.testing.assert_close(cached[index]["image"], expected, rtol=0, atol=0)
            rows[0].path.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "source changed"):
                ImageCache(root / "cache", rows, 32)

    def test_age_statistics_and_flags(self):
        stats = AgeStats.fit([20, 40, 60, 0, 137, float("nan")])
        self.assertEqual(stats.median, 40)
        self.assertEqual(stats.mean, 40)
        self.assertEqual(stats.transform(float("nan")), [0, 1, 0])
        self.assertEqual(stats.transform(137), [0, 0, 1])
        self.assertAlmostEqual(stats.transform(60)[0], np.sqrt(1.5))
        with self.assertRaises(ValueError):
            AgeStats.fit([0, float("nan")])

    def test_case_mapping_csv_order_and_submission_transforms(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root)
            rows = read_samples(root, "test", False)
            self.assertEqual(rows[0].name, "test_002.jpg")
            self.assertEqual(rows[0].path.suffix, ".JPG")
            dataset = FundusDataset(rows, AgeStats(50, 50, 10), build_transform(32))
            first = dataset[0]
            self.assertEqual(first["image"].shape, (3, 32, 32))
            self.assertTrue(torch.equal(first["image"], dataset[0]["image"]))
            self.assertNotIn("label", first)
            self.assertLessEqual(first["image"].max(), 1)
            self.assertGreaterEqual(first["image"].min(), -1)
            rows[0].path.unlink()
            with self.assertRaises(FileNotFoundError):
                read_samples(root, "test", False)

    def test_case_collision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root)
            path = root / "train" / "train_000.jpg"
            if path.exists():
                self.skipTest("Case-insensitive filesystem cannot represent this collision")
            Image.new("RGB", (8, 8)).save(path)
            with self.assertRaisesRegex(ValueError, "Ambiguous"):
                read_samples(root, "train")

    def test_audit_reports_bad_images_and_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root)
            rows = read_samples(root, "train")
            rows[1].path.write_bytes(rows[0].path.read_bytes())
            rows[2].path.write_bytes(b"not a jpeg")
            report = audit_samples({"train": rows}, hashes=True)
            self.assertEqual(len(report["errors"]), 1)
            self.assertEqual(len(report["duplicate_groups"]), 1)


class NumericTests(unittest.TestCase):
    def test_config_and_metrics(self):
        with self.assertRaises(TypeError):
            Config.from_dict({"train": {"learning_rtae": 0.01}})
        with self.assertRaises(ValueError):
            Config.from_dict({"train": {"epochs": 1}})
        metrics = binary_metrics([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8])
        self.assertEqual(metrics["auroc"], 1)
        self.assertEqual(metrics["ap"], 1)
        self.assertEqual(metrics["auprc_trapz"], 1)
        self.assertIsNone(binary_metrics([0, 0], [0.1, 0.2])["auroc"])
        with self.assertRaises(ValueError):
            binary_metrics([0, 1], [float("nan"), 0.2])

    def test_schedule_endpoints(self):
        config = TrainConfig()
        self.assertEqual(learning_rate(0, 10, 2, config), config.learning_rate / 2)
        self.assertEqual(learning_rate(1, 10, 2, config), config.learning_rate)
        self.assertEqual(learning_rate(9, 10, 2, config), config.min_learning_rate)

    def test_accumulation_matches_full_batch_including_short_windows(self):
        class Samples(Dataset):
            def __len__(self):
                return 7

            def __getitem__(self, index):
                return {"image": torch.tensor([index / 10, 0.5]), "age": torch.zeros(3),
                        "label": torch.tensor(float(index % 2))}

        class Linear(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(2, 1)

            def forward(self, image, age):
                return self.layer(image).squeeze(1)

        for micro, accumulation, full in ((3, 3, 7), (2, 2, 4)):
            torch.manual_seed(7)
            left = Linear()
            right = copy.deepcopy(left)
            for model, batch_size, steps in ((left, micro, accumulation), (right, full, 1)):
                config = TrainConfig(accumulation_steps=steps, learning_rate=0.1,
                                     min_learning_rate=0.1, grad_clip=100, amp=False)
                optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
                loader = DataLoader(Samples(), batch_size=batch_size)
                train_one_epoch(model, loader, optimizer, torch.amp.GradScaler("cuda", enabled=False),
                                torch.device("cpu"), config, 0, 2, 0)
            for a, b in zip(left.parameters(), right.parameters()):
                torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-6)


class ModelTests(unittest.TestCase):
    def test_real_convnext_both_variants_and_backward(self):
        torch.manual_seed(42)
        b0 = build_model(ModelConfig(params={"use_age": False})).eval()
        torch.manual_seed(42)
        b1 = build_model(ModelConfig(params={"use_age": True})).eval()
        for key, value in b0.encoder.state_dict().items():
            self.assertTrue(torch.equal(value, b1.encoder.state_dict()[key]))
        images, ages = torch.randn(2, 3, 32, 32), torch.randn(2, 3)
        with torch.no_grad():
            torch.testing.assert_close(b0(images, ages), b0(images, ages + 10), rtol=0, atol=0)
        logits = b1(images, ages)
        self.assertEqual(logits.shape, (2,))
        nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([0., 1.])).backward()
        self.assertTrue(torch.isfinite(b1.age_encoder[0].weight.grad).all())
        self.assertTrue(torch.isfinite(b1.encoder[0][0][0].weight.grad).all())


class WorkflowTests(unittest.TestCase):
    def test_train_resume_evaluate_predict_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root / "data")
            config = Config.from_dict({
                "device": "cpu", "output_root": str(root / "outputs"),
                "data": {"root": str(root / "data"), "image_size": 32, "num_workers": 0},
                "model": {"name": "_test_tiny", "params": {}},
                "train": {"epochs": 3, "warmup_epochs": 0, "batch_size": 2,
                          "accumulation_steps": 2, "learning_rate": 0.01, "amp": False}})
            full = train(config)
            partial = train(config, stop_after_epochs=1)
            prepare_cache(root / "data", root / "cache", 32, 1)
            resumed = train(config, resume=partial / "last.pt", image_cache=root / "cache")
            a = read_checkpoint(full / "last.pt", "training")
            b = read_checkpoint(resumed / "last.pt", "training")
            self.assertEqual(a["global_step"], 9)
            self.assertEqual(a["global_step"], b["global_step"])
            self.assertLess(a["history"][-1]["train"]["loss"], a["history"][0]["train"]["loss"])
            for key, value in a["model_state"].items():
                torch.testing.assert_close(value, b["model_state"][key], rtol=0, atol=0)
            model, loaded, age = load_model(full / "best.pt", torch.device("cpu"))
            samples = read_samples(root / "data", "valid")
            loader = make_loader(FundusDataset(samples, age, build_transform(32)), 2, 0, 42)
            predictions, metrics = collect_predictions(model, loader, torch.device("cpu"))
            self.assertEqual(metrics["count"], 4)
            saved_metrics = json.loads((full / "metrics.json").read_text())["valid"]
            self.assertEqual(metrics, saved_metrics)
            main(["evaluate", "--checkpoint", str(full / "best.pt"), "--data-root", str(root / "data"),
                  "--device", "cpu", "--num-workers", "0", "--output-dir", str(root / "eval")])
            main(["predict", "--checkpoint", str(full / "best.pt"), "--data-root", str(root / "data"),
                  "--device", "cpu", "--num-workers", "0", "--output", str(root / "submission.csv")])
            with (root / "submission.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([r["Image_name"] for r in rows], ["test_002.jpg", "test_001.jpg", "test_000.jpg"])
            self.assertTrue(all(0 <= float(row["Probability"]) <= 1 for row in rows))
            changed = Config.from_dict(config.to_dict())
            changed.train.pos_weight = 30
            with self.assertRaisesRegex(ValueError, "config differs"):
                train(changed, resume=partial / "last.pt")


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
