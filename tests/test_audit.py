"""Audit evidence must retain old epochs, survive resume and detect changed bytes."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest

import torch

from test_baseline import fixture
from glaucoma.cli import main
from glaucoma.config import Config
from glaucoma.engine.audit import save_epoch_checkpoint, verify_run_audit
from glaucoma.engine.checkpoint import read_checkpoint
from glaucoma.engine.trainer import train
from glaucoma.utils import sha256_file


class AuditTests(unittest.TestCase):
    def config(self, root):
        return Config.from_dict({"device": "cpu", "output_root": str(root / "outputs"),
            "data": {"root": str(root / "data"), "image_size": 32, "num_workers": 0},
            "model": {"name": "_test_tiny", "params": {}},
            "train": {"epochs": 3, "warmup_epochs": 0, "batch_size": 3, "amp": False}})

    def test_epoch_files_initialization_archive_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root / "data")
            run = train(self.config(root))
            result = verify_run_audit(run)
            self.assertTrue(result["training_complete"])
            self.assertEqual(result["recorded_epochs"], [1, 2, 3])
            ledger = json.loads((run / "checkpoint_manifest.json").read_text())
            for row in ledger["epochs"]:
                path = run / row["file"]
                self.assertEqual(row["sha256"], sha256_file(path))
                self.assertEqual(row["bytes"], path.stat().st_size)
                self.assertTrue(row["saved_at_utc"].endswith("+00:00"))
                self.assertEqual(read_checkpoint(path, "training")["epoch"], row["epoch"] - 1)
            self.assertEqual(sha256_file(run / "last.pt"), ledger["epochs"][-1]["sha256"])
            initial = json.loads((run / "initialization.json").read_text())
            self.assertEqual(initial["mode"], "new")
            self.assertEqual(initial["model_initial_sha256"], sha256_file(run / initial["model_initial"]))
            with tarfile.open(run / "source.tar.gz") as archive:
                source = archive.extractfile("src/glaucoma/engine/trainer.py").read()
                self.assertIn(b"save_epoch_checkpoint", source)
            main(["verify-run", "--run-dir", str(run)])
            payload = read_checkpoint(run / "last.pt", "training")
            with self.assertRaisesRegex(FileExistsError, "must not be overwritten"):
                save_epoch_checkpoint(run, payload)
            with (run / ledger["epochs"][0]["file"]).open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                verify_run_audit(run)

    def test_resume_keeps_source_checkpoint_and_parent_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root / "data")
            config = self.config(root)
            partial = train(config, stop_after_epochs=1)
            self.assertFalse(verify_run_audit(partial)["training_complete"])
            original = sha256_file(partial / "checkpoints/epoch_0001.pt")
            resumed = train(config, resume=partial / "last.pt")
            initial = json.loads((resumed / "initialization.json").read_text())
            self.assertEqual(initial["mode"], "resume")
            self.assertEqual(initial["parent_run_directory"], str(partial.resolve()))
            self.assertEqual(initial["source"]["sha256"], original)
            self.assertEqual(sha256_file(resumed / initial["source"]["file"]), original)
            self.assertEqual(sha256_file(partial / "checkpoints/epoch_0001.pt"), original)
            self.assertEqual(verify_run_audit(resumed)["recorded_epochs"], [2, 3])
            payload = read_checkpoint(resumed / "last.pt", "training")
            self.assertEqual(len(payload["history"]), 3)
            self.assertIn("optimizer", payload)
            self.assertIn("scaler", payload)
            self.assertIn("rng", payload)

    def test_launcher_keeps_failed_entrypoint_log_and_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "invalid.json"
            config.write_text('{"train":{"epochs":0}}')
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2")
            completed = subprocess.run([sys.executable, "scripts/train_audited.py", "--config", str(config),
                "--data-root", str(root), "--device", "cpu", "--output-root", str(root / "out")],
                capture_output=True, text=True, env=environment, timeout=45)
            self.assertNotEqual(completed.returncode, 0)
            launch = next((root / "out/launches").glob("*/launch.json"))
            record = json.loads(launch.read_text())
            self.assertEqual(record["exit_code"], completed.returncode)
            self.assertIn("ValueError", Path(record["log"]).read_text())
            self.assertEqual(record["train_log_sha256"], sha256_file(Path(record["log"])))
            self.assertEqual((launch.parent / "train.exit").read_text().strip(), str(completed.returncode))


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
