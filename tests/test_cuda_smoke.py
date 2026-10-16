"""Opt-in GPU smoke test. Set CUDA_VISIBLE_DEVICES to an allocated idle GPU."""

import os
import unittest

import torch
from torch.utils.data import DataLoader, Dataset

from glaucoma.config import ModelConfig, TrainConfig
from glaucoma.engine.optimization import build_optimizer, train_one_epoch
from glaucoma.models import build_model


class SyntheticFundus(Dataset):
    def __len__(self):
        return 16

    def __getitem__(self, index):
        return {"image": torch.randn(3, 384, 384, generator=torch.Generator().manual_seed(index)),
                "age": torch.tensor([index / 16, 0., 0.]), "label": torch.tensor(float(index % 2))}


@unittest.skipUnless(os.environ.get("GLAUCOMA_CUDA_SMOKE") == "1" and torch.cuda.is_available(),
                     "Opt-in CUDA test; requires GLAUCOMA_CUDA_SMOKE=1")
class CudaSmokeTests(unittest.TestCase):
    def test_both_variants_fp16_at_default_resolution_and_batch(self):
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        config = TrainConfig(accumulation_steps=1)
        for use_age in (False, True):
            with self.subTest(use_age=use_age):
                torch.manual_seed(42)
                model = build_model(ModelConfig(params={"use_age": use_age})).to(device)
                optimizer = build_optimizer(model, config)
                scaler = torch.amp.GradScaler("cuda")
                loader = DataLoader(SyntheticFundus(), batch_size=16)
                torch.cuda.reset_peak_memory_stats(device)
                step = 0
                # GradScaler may need to lower its initial scale before the first update.
                for _ in range(4):
                    metrics, step = train_one_epoch(model, loader, optimizer, scaler, device, config, step, 10, 2)
                    if step:
                        break
                self.assertEqual(step, 1)
                self.assertTrue(all(torch.isfinite(p).all().item() for p in model.parameters()))
                print(f"CUDA smoke use_age={use_age}: step={step}, "
                      f"peak_allocated_MiB={torch.cuda.max_memory_allocated(device) / 1024**2:.1f}, "
                      f"scale={scaler.get_scale()}, loss={metrics['loss']:.6f}", flush=True)
                del model, optimizer, scaler
                torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
