"""Decode-policy, parity and latency-harness tests (synthetic images, CPU only)."""

import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
from torch import nn

from glaucoma.config import Config, ModelConfig
from glaucoma.data.age import AgeStats
from glaucoma.data.cache import ImageCache, prepare_cache
from glaucoma.data.dataset import FundusDataset, load_rgb, read_samples
from glaucoma.data.transforms import build_transform
from glaucoma.engine.checkpoint import inference_payload, load_model, save_checkpoint
from glaucoma.engine.trainer import _resume_compatible
from glaucoma.models import build_model, register_model
from glaucoma.utils import percentile

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("benchmark_latency", ROOT / "scripts/benchmark_latency.py")
benchmark_latency = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark_latency)


@register_model("_test_latency_tiny")
class TinyLatencyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(6, 1)

    def forward(self, image, age):
        pooled = image.mean(dim=(2, 3))
        return self.head(torch.cat((pooled, age), dim=1)).squeeze(1)


def smooth_jpeg(path, size, seed=3):
    """Low-frequency fundus-like image: draft and full decoding should agree."""
    width, height = size
    yy, xx = np.mgrid[0:height, 0:width]
    field = (80 + 120 * np.exp(-(((xx - width * 0.5) / (width * 0.45)) ** 2
                                + ((yy - height * 0.5) / (height * 0.45)) ** 2)))
    noise = np.random.default_rng(seed).normal(0, 1.5, (height, width))
    pixels = np.clip(field + noise, 0, 255).astype(np.uint8)
    image = Image.fromarray(np.dstack([pixels, np.clip(pixels * 0.6, 0, 255).astype(np.uint8),
                                       np.clip(pixels * 0.3, 0, 255).astype(np.uint8)]))
    image.save(path, quality=95)
    return image


def fixture(root, sizes=((1200, 800), (640, 480), (300, 200))):
    for split, count in (("train", 3), ("valid", 3), ("test", 2)):
        directory = root / split
        directory.mkdir(parents=True)
        with (root / f"{split}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Image_name", "Age"] + ([] if split == "test" else ["True_Label"]))
            for index in range(count):
                name = f"{split}_{index:03d}.jpg"
                smooth_jpeg(directory / name, sizes[index % len(sizes)])
                writer.writerow([name, 40 + index] + ([] if split == "test" else [index % 2]))


class DecodeTests(unittest.TestCase):
    def test_draft_never_upscales_and_only_affects_jpeg(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jpeg = root / "large.jpg"
            smooth_jpeg(jpeg, (1200, 800))
            draft = load_rgb(jpeg, "draft", 384)
            self.assertEqual(draft.mode, "RGB")
            self.assertTrue(384 <= draft.width <= 768 and 384 <= draft.height <= 768, draft.size)
            full = load_rgb(jpeg, "full", 384)
            self.assertEqual(full.size, (1200, 800))
            self.assertEqual(load_rgb(jpeg).size, full.size)

            png = root / "image.png"
            Image.new("RGB", (900, 500), (10, 120, 200)).save(png)
            self.assertEqual(load_rgb(png, "draft", 384).size, (900, 500))
            with self.assertRaises(ValueError):
                load_rgb(jpeg, "draft2x", 384)

    def test_draft_full_tensor_parity_on_smooth_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root)
            rows = read_samples(root, "valid")
            transform = build_transform(64, preprocessed=False)
            for row in rows:
                full = transform(load_rgb(row.path, "full", 64))
                draft = transform(load_rgb(row.path, "draft", 64))
                self.assertEqual(full.shape, draft.shape)
                self.assertLessEqual(float((full - draft).abs().mean()), 0.02)
                self.assertTrue(torch.isfinite(draft).all())

    def test_decode_policy_config_and_checkpoint_round_trip(self):
        config = Config.from_dict({"data": {"root": "dataset"}})
        self.assertEqual(config.data.decode, "full")
        with self.assertRaisesRegex(ValueError, "decode"):
            Config.from_dict({"data": {"decode": "bilinear"}})
        # Old payloads without the field keep the historical default.
        with tempfile.TemporaryDirectory() as directory:
            model = build_model(ModelConfig(name="_test_latency_tiny", params={}))
            config.model = ModelConfig(name="_test_latency_tiny", params={})
            config.data.decode = "draft"
            path = Path(directory) / "best.pt"
            payload = inference_payload(model, config, AgeStats(50, 50, 10), 0, {"auroc": 0.5})
            save_checkpoint(path, payload)
            loaded, loaded_config, _ = load_model(path, torch.device("cpu"))
            self.assertEqual(loaded_config.data.decode, "draft")
            self.assertIsInstance(loaded, TinyLatencyModel)
        # Resume must reject a decode-policy change like any other numeric setting.
        saved = {"data": {"decode": "full"}}
        changed = Config.from_dict({"data": {"decode": "draft"}})
        with self.assertRaisesRegex(ValueError, "Resume config differs"):
            _resume_compatible(changed, saved)

    def test_dataset_threads_decode_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root)
            rows = read_samples(root, "valid")
            stats = AgeStats(50, 50, 10)
            full = FundusDataset(rows, stats, build_transform(32), decode="full", image_size=32)
            draft = FundusDataset(rows, stats, build_transform(32), decode="draft", image_size=32)
            self.assertEqual(full[0]["image"].shape, draft[0]["image"].shape)
            self.assertEqual(full.decode, "full")
            self.assertEqual(draft.decode, "draft")


class CachePolicyTests(unittest.TestCase):
    def test_manifest_v3_decode_gate_and_legacy_v2(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root / "data")
            prepare_cache(root / "data", root / "cache", 32, 1)
            manifest_path = root / "cache" / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(manifest["version"], 3)
            self.assertEqual(manifest["decode"], "full")
            self.assertEqual(manifest["preprocessing"]["decode"], "full")
            rows = read_samples(root / "data", "train")
            ImageCache(root / "cache", rows, 32)
            with self.assertRaisesRegex(ValueError, "decode mismatch"):
                ImageCache(root / "cache", rows, 32, decode="draft")

            # A legacy v2 manifest has no decode field and must still mean full.
            manifest["version"] = 2
            manifest.pop("decode")
            manifest["preprocessing"].pop("decode")
            manifest_path.write_text(json.dumps(manifest))
            ImageCache(root / "cache", rows, 32)
            with self.assertRaisesRegex(ValueError, "decode mismatch"):
                ImageCache(root / "cache", rows, 32, decode="draft")

    def test_draft_cache_records_and_requires_draft(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root / "data")
            prepare_cache(root / "data", root / "cache", 32, 1, decode="draft")
            manifest = json.loads((root / "cache" / "manifest.json").read_text())
            self.assertEqual((manifest["version"], manifest["decode"]), (3, "draft"))
            rows = read_samples(root / "data", "valid")
            ImageCache(root / "cache", rows, 32, decode="draft")
            with self.assertRaisesRegex(ValueError, "decode mismatch"):
                ImageCache(root / "cache", rows, 32, decode="full")


class HarnessTests(unittest.TestCase):
    def test_percentile_matches_linear_interpolation(self):
        values = [5.0, 1.0, 3.0, 2.0, 4.0]
        self.assertEqual(percentile(values, 0), 1.0)
        self.assertEqual(percentile(values, 50), 3.0)
        self.assertEqual(percentile(values, 100), 5.0)
        self.assertAlmostEqual(percentile(values, 25), 2.0)
        with self.assertRaises(ValueError):
            percentile(values, 101)

    def test_cpu_only_benchmark_on_tiny_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root / "data")
            config = Config.from_dict({
                "device": "cpu",
                "data": {"root": str(root / "data"), "image_size": 32,
                         "num_workers": 0, "crop_black_border": False},
                "model": {"name": "_test_latency_tiny", "params": {}},
                "train": {"amp": False}})
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config.to_dict()))
            output = root / "latency.json"
            code = benchmark_latency.main([
                "--data-root", str(root / "data"), "--config", str(config_path), "--split", "valid",
                "--cpu-only", "--images", "3", "--warmup", "1", "--no-enforce",
                "--output", str(output)])
            self.assertEqual(code, 0)
            report = json.loads(output.read_text())
            self.assertEqual(report["precision"], "fp32")
            self.assertEqual(report["decode"], "full")
            self.assertEqual(len(report["records"]), 3)
            self.assertIn("total", report["summary_ms"])
            self.assertGreater(report["summary_ms"]["total"]["p50"], 0)
            self.assertIn("latency_pass", report["gates"])
            self.assertIsNone(report["gates"]["model_size_pass"])

    def test_benchmark_rejects_fp16_on_cpu(self):
        with self.assertRaisesRegex(ValueError, "fp16"):
            benchmark_latency.main(["--data-root", "/nonexistent", "--cpu-only",
                                    "--precision", "fp16", "--images", "1"])


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
