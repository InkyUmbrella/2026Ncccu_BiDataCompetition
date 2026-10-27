"""Synthetic/offline DINOv2 checks. No real data or implicit model downloads."""

import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from glaucoma.config import Config
from glaucoma.data.age import AgeStats
from glaucoma.engine.checkpoint import inference_payload, load_model, save_checkpoint
from glaucoma.engine.optimization import build_optimizer
from glaucoma.models import build_model
from glaucoma.models.dinov2 import DinoSmallBackbone
from glaucoma.models.pretrained import DINOV2_REVISION, DINOV2_URL, dinov2_source
from glaucoma.utils import sha256_file


class DinoTests(unittest.TestCase):
    def config(self):
        return Config.load("configs/experiments/dinov2_vits14_crop392_ema.json")

    def test_config_and_optimizer_preserve_recipe_and_reject_bad_shape(self):
        config = self.config()
        self.assertEqual(config.train.batch_size * config.train.accumulation_steps, 128)
        bad = config.to_dict()
        bad["data"]["image_size"] = 384
        with self.assertRaisesRegex(ValueError, "multiple of 14"):
            Config.from_dict(bad)
        bad = config.to_dict()
        bad["model"]["weights"] = "IMAGENET1K_V1"
        with self.assertRaisesRegex(ValueError, "weights must be"):
            Config.from_dict(bad)
        model = build_model(config.model, load_pretrained=False)
        optimizer = build_optimizer(model, config.train)
        rates = {id(parameter): group["lr"] for group in optimizer.param_groups for parameter in group["params"]}
        for name, parameter in model.named_parameters():
            self.assertAlmostEqual(rates[id(parameter)], 1e-5 if name.startswith("encoder.") else 1e-4)

    def test_official_state_layout_forward_gradients_and_offline_best(self):
        config = self.config()
        model = build_model(config.model, load_pretrained=False)
        state = model.encoder.state_dict()
        self.assertEqual(state["pos_embed"].shape, (1, 1370, 384))
        self.assertEqual(state["blocks.11.attn.qkv.weight"].shape, (1152, 384))
        self.assertEqual(state["blocks.0.ls1.gamma"].shape, (384,))
        image, age = torch.randn(2, 3, 42, 42), torch.zeros(2, 3)
        model.train()
        output = model(image, age)
        self.assertEqual(output.shape, (2,))
        output.sum().backward()
        self.assertIsNotNone(model.encoder.patch_embed.proj.weight.grad)
        self.assertIsNotNone(model.head[-1].weight.grad)
        with self.assertRaisesRegex(ValueError, "multiples"):
            model(torch.randn(1, 3, 40, 40), age[:1])
        model.eval()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "best.pt"
            save_checkpoint(checkpoint, inference_payload(model, config, AgeStats(50, 50, 10), 0, {}))
            with patch("glaucoma.models.dinov2.dinov2_source", side_effect=AssertionError("source used on inference")):
                restored, _, _ = load_model(checkpoint, torch.device("cpu"))
            with torch.inference_mode():
                torch.testing.assert_close(restored(image, age), model(image, age), atol=0, rtol=0)

    def test_pretrained_loading_is_strict_and_uses_only_encoder(self):
        config = self.config()
        donor = DinoSmallBackbone().state_dict()
        donor["patch_embed.proj.bias"].fill_(0.125)
        with patch("glaucoma.models.dinov2.dinov2_source", return_value=(Path("synthetic.pth"), None, None)), \
             patch("glaucoma.models.dinov2.torch.load", return_value=donor) as loading:
            model = build_model(config.model)
        loading.assert_called_once_with(Path("synthetic.pth"), map_location="cpu", weights_only=True)
        torch.testing.assert_close(model.encoder.patch_embed.proj.bias, donor["patch_embed.proj.bias"])
        bad = dict(donor)
        bad.pop("mask_token")
        with patch("glaucoma.models.dinov2.dinov2_source", return_value=(Path("synthetic.pth"), None, None)), \
             patch("glaucoma.models.dinov2.torch.load", return_value=bad):
            with self.assertRaises(RuntimeError):
                build_model(config.model)

    def test_source_manifest_detects_weight_and_license_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "checkpoints"
            cache.mkdir()
            source = cache / "dinov2_vits14_pretrain.pth"
            source.write_bytes(b"synthetic fixture, not an official model")
            docs = cache / "dinov2_vits14_documents"
            docs.mkdir()
            (docs / "LICENSE").write_bytes(b"synthetic license fixture")
            record = {"format_version": 1, "model": "dinov2_vits14", "weights_id": "LVD142M",
                      "url": DINOV2_URL, "license": "Apache-2.0", "upstream_revision": DINOV2_REVISION,
                      "sha256": sha256_file(source), "bytes": source.stat().st_size}
            (cache / "dinov2_vits14_source.json").write_text(json.dumps(record), encoding="utf-8")
            with patch("torch.hub.get_dir", return_value=str(root)), \
                 patch("glaucoma.models.pretrained.DINOV2_DOCUMENTS", {"LICENSE": sha256_file(docs / "LICENSE")}):
                self.assertEqual(dinov2_source()[0], source)
                source.write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "SHA256"):
                    dinov2_source()

    def test_fixed_512_initialization_cannot_start_randomly(self):
        from glaucoma.engine.trainer import train
        config = Config.load("configs/experiments/b0_crop512_refine_ema.json")
        with self.assertRaisesRegex(ValueError, "configured SHA256"):
            train(config)

    @unittest.skipUnless(os.environ.get("GLAUCOMA_DINOV2_UPSTREAM"), "Opt-in pinned upstream architecture parity")
    def test_matches_pinned_upstream_features(self):
        # Caller supplies an audited local checkout; no network access is allowed here.
        repository = Path(os.environ["GLAUCOMA_DINOV2_UPSTREAM"]).resolve()
        import subprocess
        revision = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
        self.assertEqual(revision, DINOV2_REVISION)
        with patch.dict(os.environ, {"XFORMERS_DISABLED": "1"}):
            upstream = torch.hub.load(str(repository), "dinov2_vits14", source="local", pretrained=False).eval()
        ours = DinoSmallBackbone().eval()
        upstream.load_state_dict(ours.state_dict(), strict=True)
        with torch.inference_mode():
            for shape in (42, 56):
                image = torch.randn(1, 3, shape, shape)
                torch.testing.assert_close(ours(image), upstream(image), atol=1e-5, rtol=1e-5)

    @unittest.skipUnless(os.environ.get("GLAUCOMA_DINOV2_OFFICIAL") == "1", "Opt-in original cached weight loading")
    def test_original_cached_weights_load_without_network(self):
        with patch("torch.hub.load", side_effect=AssertionError("Torch Hub execution is forbidden")), \
             patch("torch.hub.download_url_to_file", side_effect=AssertionError("download during loading")):
            model = build_model(self.config().model).eval()
        with torch.inference_mode():
            result = model(torch.randn(1, 3, 42, 42), torch.zeros(1, 3))
        self.assertTrue(torch.isfinite(result).all())

    @unittest.skipUnless(os.environ.get("GLAUCOMA_DINOV2_SMOKE") == "1", "Opt-in DINOv2 CUDA/EMA/resume check")
    def test_cuda_resume_keeps_ema_and_complete_state(self):
        from test_baseline import fixture
        from glaucoma.engine.checkpoint import read_checkpoint
        from glaucoma.engine.trainer import train
        self.assertTrue(torch.cuda.is_available())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root / "data")
            config = copy.deepcopy(self.config())
            config.model.weights = None  # Synthetic engineering run, never a formal score.
            config.data.root, config.output_root = str(root / "data"), str(root / "out")
            config.data.image_size, config.data.num_workers = 42, 0
            config.train.epochs, config.train.warmup_epochs = 2, 0
            config.train.batch_size, config.train.accumulation_steps = 3, 1
            partial = train(config, stop_after_epochs=1)
            resumed = train(config, resume=partial / "last.pt")
            saved = read_checkpoint(resumed / "last.pt", "training")
            self.assertEqual(saved["epoch"], 1)
            self.assertEqual(int(saved["ema"]["n_averaged"]), saved["global_step"])
            for field in ("optimizer", "scaler", "rng", "history", "best"):
                self.assertIn(field, saved)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
