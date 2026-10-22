"""MoCo-v2 SSL tests: synthetic data only, CPU, no competition images."""

import json
import tempfile
from pathlib import Path
import unittest
from unittest import mock

from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from test_baseline import fixture

from glaucoma.config import Config, ModelConfig
from glaucoma.data.dataset import FundusDataset
from glaucoma.engine.checkpoint import (initialize_from_encoder, read_ssl_encoder)
from glaucoma.engine.trainer import train
from glaucoma.models import build_model
from glaucoma.ssl.augment import TwoViewTransform, build_ssl_transform
from glaucoma.ssl.checkpoint import (encoder_payload, load_recovery, read_encoder, read_recovery,
                                     recovery_payload, save_encoder, save_recovery)
from glaucoma.ssl.config import SSLConfig, SSLDataConfig, SSLTrainConfig
from glaucoma.ssl.model import MoCoV2, build_encoder, build_projector
from glaucoma.ssl.probe import effective_rank, feature_std, probe_metrics
from glaucoma.ssl.trainer import load_train_rows, make_ssl_loader, pretrain
from glaucoma.data.age import AgeStats
from glaucoma.data.dataset import Sample


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.GELU(),
                                 nn.AdaptiveAvgPool2d(1), nn.Flatten(1))

    def forward(self, image):
        return self.net(image)


def tiny_moco(queue_size=8, momentum=0.9):
    return MoCoV2(encoder=TinyEncoder(), projector=build_projector(4, 16, 4),
                  queue_size=queue_size, temperature=0.2, momentum=momentum)


def ssl_config(root, **train):
    values = dict(device="cpu", output_root=str(root / "outputs"),
                  data=dict(root=str(root / "data"), image_size=96, ssl_size=96,
                            num_workers=0, crop_black_border=False, min_crop_scale=0.5),
                  train=dict(epochs=1, batch_size=3, learning_rate=0.05, warmup_epochs=0,
                             queue_size=8, save_every=1, probe_every=10, amp=False))
    values["train"].update(train)
    return SSLConfig.from_dict(values)


class AugmentTests(unittest.TestCase):
    def test_two_views_shape_range_determinism_and_collation(self):
        image = Image.new("RGB", (384, 384), (90, 120, 150))
        transform = build_ssl_transform(224)
        self.assertIsInstance(transform, TwoViewTransform)

        torch.manual_seed(7)
        view1, view2 = transform(image)
        self.assertEqual(tuple(view1.shape), (3, 224, 224))
        self.assertEqual(tuple(view2.shape), (3, 224, 224))
        for view in (view1, view2):
            self.assertLessEqual(float(view.max()), 1.0)
            self.assertGreaterEqual(float(view.min()), -1.0)

        torch.manual_seed(7)
        again1, again2 = transform(image)
        torch.testing.assert_close(again1, view1, rtol=0, atol=0)
        torch.testing.assert_close(again2, view2, rtol=0, atol=0)

        self.assertEqual(image.getpixel((0, 0)), (90, 120, 150))

        class Tuples(Dataset):
            def __len__(self):
                return 4

            def __getitem__(self, index):
                return {"image": (torch.full((3, 8, 8), float(index)),
                                  torch.full((3, 8, 8), -float(index))),
                        "name": f"train_{index}"}

        batch = next(iter(DataLoader(Tuples(), batch_size=4)))
        self.assertIsInstance(batch["image"], (list, tuple))
        self.assertEqual(tuple(batch["image"][0].shape), (4, 3, 8, 8))
        self.assertEqual(float(batch["image"][1][0, 0, 0, 0]), 0.0)


class ModelTests(unittest.TestCase):
    def test_queue_fifo_wrap_and_momentum_semantics(self):
        model = tiny_moco(queue_size=8, momentum=0.9)
        self.assertFalse(any(p.requires_grad for p in model.momentum_encoder.parameters()))
        self.assertFalse(any(p.requires_grad for p in model.momentum_projector.parameters()))

        first = torch.arange(6, dtype=torch.float32).unsqueeze(1).expand(6, 4).contiguous()
        model.dequeue_and_enqueue(first)
        self.assertEqual(int(model.queue_ptr.item()), 6)
        torch.testing.assert_close(model.queue[:6], first)
        self.assertEqual(model.queue_fill, 6)

        second = (torch.arange(10, 14, dtype=torch.float32).unsqueeze(1).expand(4, 4)).contiguous()
        model.dequeue_and_enqueue(second)
        self.assertEqual(int(model.queue_ptr.item()), 2)
        torch.testing.assert_close(model.queue[6:8], second[:2])
        torch.testing.assert_close(model.queue[0:2], second[2:])
        torch.testing.assert_close(model.queue[2:6], first[2:])
        self.assertEqual(model.queue_fill, 8)

        with self.assertRaises(ValueError):
            model.dequeue_and_enqueue(torch.zeros(9, 4))

        target_before = model.momentum_encoder.net[0].weight.detach().clone()
        with torch.no_grad():
            model.encoder.net[0].weight.add_(1.0)
        online = model.encoder.net[0].weight.detach().clone()
        model.momentum_update()
        torch.testing.assert_close(model.momentum_encoder.net[0].weight,
                                   0.9 * target_before + 0.1 * online)

    def test_forward_shapes_and_finite_loss(self):
        model = tiny_moco(queue_size=8)
        images = torch.randn(3, 3, 8, 8)
        logits, targets, keys = model(images, images, return_keys=True)
        self.assertEqual(tuple(logits.shape), (3, 9))
        self.assertTrue(torch.equal(targets, torch.zeros(3, dtype=torch.long)))
        self.assertEqual(tuple(keys.shape), (3, 4))
        self.assertTrue(torch.isfinite(F.cross_entropy(logits, targets)))
        # Two-argument form keeps the documented tuple contract.
        pair = model(images, images)
        self.assertEqual(len(pair), 2)
        self.assertEqual(tuple(pair[0].shape), (3, 9))

    def test_infonce_loss_decreases_on_separable_views(self):
        torch.manual_seed(3)
        # A small queue keeps the negatives fresh instead of cycling the same
        # keys; the loss scale still depends on momentum, so compare windows
        # after the queue has filled rather than the empty-queue transient.
        model = tiny_moco(queue_size=8, momentum=0.9)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        pool = torch.randn(64, 3, 8, 8)
        losses = []
        for _ in range(100):
            model.train()
            images = pool[torch.randperm(len(pool))[:8]]
            optimizer.zero_grad(set_to_none=True)
            logits, targets, keys = model(images, images, return_keys=True)
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()
            model.momentum_update()
            model.dequeue_and_enqueue(keys)
            losses.append(float(loss))
        self.assertLess(sum(losses[-5:]) / 5, sum(losses[8:13]) / 5)
        self.assertLess(losses[-1], losses[8])

    def test_real_encoder_contract(self):
        encoder = build_encoder()
        with torch.no_grad():
            output = encoder(torch.randn(2, 3, 32, 32))
        self.assertEqual(tuple(output.shape), (2, 768))


class CheckpointTests(unittest.TestCase):
    def test_encoder_payload_round_trip_and_strictness(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = SSLConfig()
            source_encoder = build_encoder()
            model = MoCoV2(encoder=source_encoder, projector=build_projector(768, 32, 8),
                           queue_size=4)
            payload = encoder_payload(model, config, {"data": {"split": "train"},
                                                      "augmentation": "unit-test"}, 3, {"loss": 1.0})
            self.assertEqual(payload["kind"], "ssl_encoder")
            self.assertNotIn("projector", json.dumps(sorted(payload)))
            path = root / "encoder.pt"
            save_encoder(path, payload)
            read = read_ssl_encoder(path)
            self.assertEqual(read["model"]["name"], "convnext_tiny")

            target = build_model(ModelConfig(params={"use_age": False}))
            head_before = {k: v.detach().clone() for k, v in target.head.state_dict().items()}
            initialize_from_encoder(target, read)
            for key, value in read["encoder_state"].items():
                torch.testing.assert_close(target.encoder.state_dict()[key], value, rtol=0, atol=0)
            for key, value in head_before.items():
                torch.testing.assert_close(target.head.state_dict()[key], value, rtol=0, atol=0)

            bad = dict(read)
            bad["model"] = {"name": "resnet50", "params": {}}
            with self.assertRaisesRegex(ValueError, "convnext_tiny"):
                initialize_from_encoder(target, bad)
            mismatched = tiny_moco()
            with self.assertRaisesRegex(ValueError, "does not match"):
                initialize_from_encoder(mismatched, read)

            broken = dict(read, format_version=2)
            broken_path = root / "broken.pt"
            save_encoder(broken_path, broken)
            with self.assertRaises(ValueError):
                read_ssl_encoder(broken_path)

    def test_recovery_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = tiny_moco(queue_size=8)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            scaler = torch.amp.GradScaler("cuda", enabled=False)
            with torch.no_grad():
                model.queue[0, 0] = 3.0
                model.queue_ptr[0] = 3
                model._enqueued[0] = 5
            payload = recovery_payload(model, optimizer, scaler, SSLConfig(), 2, [{"epoch": 0}],
                                       {"probes": []}, {"python": (), "numpy": (), "torch": torch.get_rng_state(),
                                                        "cuda": None}, torch.device("cpu"))
            path = root / "last.pt"
            save_recovery(path, payload)
            read = read_recovery(path)
            restored = tiny_moco(queue_size=8)
            restored_optimizer = torch.optim.SGD(restored.parameters(), lr=0.1)
            load_recovery(restored, restored_optimizer, scaler, read)
            torch.testing.assert_close(restored.queue, model.queue)
            self.assertEqual(int(restored.queue_ptr.item()), 3)
            self.assertEqual(restored.queue_fill, 5)
            for key, value in model.encoder.state_dict().items():
                torch.testing.assert_close(restored.encoder.state_dict()[key], value, rtol=0, atol=0)
            self.assertEqual(restored_optimizer.state_dict()["param_groups"][0]["lr"], 0.1)
            broken = dict(read, kind="training")
            broken_path = root / "broken.pt"
            save_recovery(broken_path, broken)
            with self.assertRaises(ValueError):
                read_recovery(broken_path)


class ProbeTests(unittest.TestCase):
    def test_separable_features_probe_and_collapse_monitors(self):
        torch.manual_seed(0)
        negatives = torch.randn(40, 8) * 0.1
        positives = torch.randn(40, 8) * 0.1 + 3.0
        train_x = torch.cat((negatives, positives)).numpy()
        train_y = torch.cat((torch.zeros(40), torch.ones(40))).long().numpy()
        metrics = probe_metrics(train_x, train_y, train_x, train_y)
        self.assertEqual(metrics["auroc"], 1.0)
        self.assertGreater(effective_rank(train_x), 1.0)
        self.assertGreater(feature_std(train_x), 0.0)
        with self.assertRaises(ValueError):
            effective_rank(train_x[:1])


class ConfigTests(unittest.TestCase):
    def test_validate_and_json_round_trip(self):
        config = SSLConfig()
        self.assertEqual(SSLConfig.from_dict(config.to_dict()).to_dict(), config.to_dict())
        for bad in ({"train": {"temperature": 0}}, {"train": {"temperature": 1}},
                    {"train": {"moco_momentum": 0}}, {"train": {"queue_size": 0}},
                    {"train": {"epochs": 1, "warmup_epochs": 1}},
                    {"data": {"ssl_size": 64}}, {"data": {"ssl_size": 385}},
                    {"data": {"decode": "fast"}}, {"train": {"batch_size": 0}}):
            with self.assertRaises(ValueError):
                SSLConfig.from_dict(bad)

    def test_shipped_configs_load(self):
        root = Path(__file__).resolve().parents[1]
        # The 200ep config uses batch 256; the shorter 100ep config was lowered
        # to 128 to fit a 16 GB V100 (commit "ssl: batch 128 / lr 0.015").
        for name, batch_size in (("moco_v2_224_200ep.json", 256),
                                 ("moco_v2_224_100ep.json", 128)):
            config = SSLConfig.load(root / "configs/ssl" / name)
            self.assertEqual(config.data.ssl_size, 224)
            self.assertEqual(config.train.batch_size, batch_size)


class PretrainTests(unittest.TestCase):
    def test_pretrain_checkpoint_resume_and_train_only_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root / "data")
            config = ssl_config(root, epochs=2, batch_size=3)

            rows = load_train_rows(config)
            self.assertTrue(all(row.path.parent.name == "train" for row in rows))
            dataset = FundusDataset(rows, AgeStats(50, 50, 10),
                                    build_ssl_transform(96, min_crop_scale=0.5))
            loader = make_ssl_loader(dataset, 3, 0, config.seed, 0)
            for batch in loader:
                self.assertTrue(all(name.startswith("train_") for name in batch["name"]))

            first = pretrain(config, stop_after_epochs=1)
            for name in ("config.json", "metadata.json", "history.json",
                         "encoder_last.pt", "last.pt", "manifest.json"):
                self.assertTrue((first / name).exists(), name)
            encoder = read_ssl_encoder(first / "encoder_last.pt")
            self.assertEqual(encoder["data"]["split"], "train")
            self.assertEqual(encoder["data"]["count"], 9)
            history = json.loads((first / "history.json").read_text())
            self.assertEqual(len(history), 1)
            self.assertTrue(0.0 < history[0]["train"]["loss"] < 100.0)

            resumed = pretrain(config, resume=first / "last.pt")
            resumed_history = json.loads((resumed / "history.json").read_text())
            self.assertEqual([row["epoch"] for row in resumed_history], [0, 1])
            self.assertEqual(read_ssl_encoder(resumed / "encoder_last.pt")["epoch"], 1)

    def test_refuses_valid_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root / "data")
            config = ssl_config(root)
            valid_row = Sample("valid_000", root / "data" / "valid" / "valid_000.JPG", 50, None)
            with mock.patch("glaucoma.ssl.trainer.read_samples", return_value=[valid_row]):
                with self.assertRaisesRegex(ValueError, "only read the train split"):
                    pretrain(config)


class InitEncoderTrainTests(unittest.TestCase):
    def test_train_init_encoder_records_hash_and_rejects_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root / "data")
            source = build_encoder()
            model = MoCoV2(encoder=source, projector=build_projector(768, 16, 8), queue_size=4)
            payload = encoder_payload(model, SSLConfig(),
                                      {"data": {"split": "train"}, "augmentation": "unit-test"}, 0, {})
            encoder_path = root / "encoder.pt"
            save_encoder(encoder_path, payload)

            config = Config.from_dict({
                "device": "cpu", "output_root": str(root / "outputs"),
                "data": {"root": str(root / "data"), "image_size": 32, "num_workers": 0},
                "model": {"name": "convnext_tiny", "params": {"use_age": False}},
                "train": {"epochs": 1, "warmup_epochs": 0, "batch_size": 3,
                          "accumulation_steps": 1, "learning_rate": 0.001,
                          "min_learning_rate": 1e-6, "grad_clip": 1.0, "amp": False}})
            run = train(config, init_encoder=encoder_path, stop_after_epochs=1)
            metadata = json.loads((run / "metadata.json").read_text())
            self.assertEqual(metadata["initialized_from_encoder"], str(encoder_path.resolve()))
            self.assertEqual(metadata["initialized_from_encoder_sha256"],
                             payload_sha256(encoder_path))

            with self.assertRaises(ValueError):
                train(config, resume=run / "last.pt", init_encoder=encoder_path)
            mismatched = dict(payload, model={"name": "resnet50", "params": {}})
            bad_path = root / "bad.pt"
            save_encoder(bad_path, mismatched)
            with self.assertRaisesRegex(ValueError, "convnext_tiny"):
                train(config, init_encoder=bad_path)


def payload_sha256(path):
    from glaucoma.utils import sha256_file
    return sha256_file(path)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
