"""One shared path must reread each image and detect CSV/spec/weight differences."""

import contextlib
import csv
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import torch
from torch import nn

from glaucoma.benchmark import picture_files, run_cost
from glaucoma.config import Config
from glaucoma.data.age import AgeStats
from glaucoma.data.dataset import Sample, load_rgb
from glaucoma.engine.checkpoint import inference_payload, save_checkpoint
from glaucoma.inference import InferenceSession, evaluate_frozen, freeze_spec, predict_samples, read_submission
from glaucoma.models import register_model


@register_model("_test_image_only")
class ImageOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Conv2d(3, 4, 1)
        self.head = nn.Linear(4, 1)

    def forward(self, image, age):
        # Intentionally asymmetric so the horizontal flip test exercises both views.
        features = self.encoder(image[:, :, :, :image.shape[-1] // 2]).mean((2, 3))
        return self.head(features).squeeze(1)


class InferenceTests(unittest.TestCase):
    def checkpoint(self, root):
        config = Config.from_dict({"device": "cpu", "data": {"image_size": 32, "num_workers": 0},
                                   "model": {"name": "_test_image_only", "params": {}}})
        path = root / "best.pt"
        torch.manual_seed(42)
        save_checkpoint(path, inference_payload(ImageOnly(), config, AgeStats(50, 50, 10), 0, {}))
        return path

    def pictures(self, root, count, start=0):
        root.mkdir()
        result = []
        for index in range(start, start + count):
            path = root / f"image_{index:03d}.png"
            image = Image.new("RGB", (36, 28), (index % 200, 20, 40))
            image.paste((220, 50, 10), (0, 0, 9, 28))
            image.save(path)
            result.append(Sample(path.name, path, float("nan"), None))
        return result

    def test_shared_prediction_is_frozen_and_hflip_averages_probabilities(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = self.checkpoint(root)
            spec = root / "inference_spec.json"
            freeze_spec(checkpoint, spec, precision="fp32", tta="hflip")
            samples = self.pictures(root / "images", 2)
            session = InferenceSession(spec, "cpu")
            probability = session.predict_image(samples[0])[0]
            image = session.transform(load_rgb(samples[0].path)).unsqueeze(0)
            with torch.inference_mode():
                expected = torch.stack([session.model(view, torch.zeros(1, 3)).sigmoid()
                                        for view in (image, image.flip(-1))]).mean().item()
            self.assertEqual(probability, expected)
            session.predict_image(samples[1])
            self.assertEqual(probability, session.predict_image(samples[0])[0])
            self.assertFalse(session.model.training)
            self.assertTrue(all(not parameter.requires_grad for parameter in session.model.parameters()))
            report = predict_samples(spec, list(reversed(samples)), root / "submission.csv", "cpu")
            self.assertEqual(report["batch_size"], 1)
            reference = read_submission(root / "submission.csv")
            self.assertEqual(list(reference), [sample.name for sample in reversed(samples)])
            with self.assertRaises(FileExistsError):
                predict_samples(spec, samples, root / "submission.csv", "cpu")

    def test_tampered_weights_or_source_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = self.checkpoint(root)
            spec = root / "spec.json"
            frozen = freeze_spec(checkpoint, spec, "fp32")
            with patch("glaucoma.inference.source_hashes", return_value={"changed": "source"}):
                with self.assertRaisesRegex(ValueError, "source changed"):
                    InferenceSession(spec, "cpu")
            with checkpoint.open("ab") as handle:
                handle.write(b"tampered")
            with self.assertRaisesRegex(ValueError, "checkpoint SHA256"):
                InferenceSession(spec, "cpu")
            self.assertEqual(frozen["threshold"], 0.5)

    def test_valid_evaluation_uses_identical_serialized_probabilities(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = root / "spec.json"
            freeze_spec(self.checkpoint(root), spec, "fp32")
            unlabelled = self.pictures(root / "valid", 4)
            samples = [Sample(row.name, row.path, row.age, index % 2) for index, row in enumerate(unlabelled)]
            result = evaluate_frozen(spec, samples, root / "evaluation", "cpu")
            records = read_submission(root / "evaluation/probabilities.csv")
            positive = [records[row.name][1] for row in samples if row.label == 1]
            negative = [records[row.name][1] for row in samples if row.label == 0]
            expected = sum(float(p > n) + 0.5 * float(p == n) for p in positive for n in negative) / 4
            self.assertEqual(result["auroc"], expected)
            self.assertEqual(result["count"], 4)
            self.assertEqual(result["positives"], 2)

    def test_cost_rereads_all_305_images_and_matches_all_three_rounds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = root / "spec.json"
            freeze_spec(self.checkpoint(root), spec, "fp32")
            samples = self.pictures(root / "measured", 100)
            self.pictures(root / "warmup", 5, 100)
            predict_samples(spec, samples, root / "submission.csv", "cpu")
            with patch("glaucoma.inference.load_rgb", wraps=load_rgb) as decoding, \
                 contextlib.redirect_stdout(io.StringIO()):
                result = run_cost(spec, root / "submission.csv", root / "measured", root / "warmup",
                                  root / "cost", "cpu", diagnostic=True)
            self.assertEqual(decoding.call_count, 305)
            self.assertEqual(result["mismatch_count"], 0)
            self.assertTrue(all(len(values) == 3 for values in result["per_image_round_ms"].values()))
            self.assertTrue(result["diagnostic"])
            self.assertFalse(result["target_gpu_software_matches"])
            rows = list(csv.reader((root / "submission.csv").open(encoding="utf-8")))
            rows[1][1] = "0"  # Detect different submitted probabilities rather than tolerating them.
            with (root / "wrong.csv").open("w", encoding="utf-8", newline="") as handle:
                csv.writer(handle).writerows(rows)
            with contextlib.redirect_stdout(io.StringIO()):
                wrong = run_cost(spec, root / "wrong.csv", root / "measured", root / "warmup",
                                 root / "wrong_cost", "cpu", diagnostic=True)
            self.assertEqual(wrong["mismatch_count"], 1)
            self.assertEqual(wrong["mismatch_rounds"][samples[0].name], [1, 2, 3])

    def test_cost_requires_exact_counts_and_distinct_warmup_names(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.pictures(root / "short", 4)
            with self.assertRaisesRegex(ValueError, "Expected 5"):
                picture_files(root / "short", 5)
            self.pictures(root / "measured", 100)
            self.pictures(root / "overlap", 5)
            with self.assertRaisesRegex(ValueError, "must be separate"):
                run_cost("not_read.json", "not_read.csv", root / "measured", root / "overlap", root / "out")


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
