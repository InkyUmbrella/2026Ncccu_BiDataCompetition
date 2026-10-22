import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, "tests")
from test_baseline import fixture  # noqa: E402

from glaucoma.config import Config  # noqa: E402
from glaucoma.engine.checkpoint import load_model, read_checkpoint  # noqa: E402
from glaucoma.engine.trainer import train  # noqa: E402

with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    fixture(root / "data")
    config = Config.from_dict({
        "device": "cuda:0", "output_root": str(root / "outputs"),
        "data": {"root": str(root / "data"), "image_size": 64, "num_workers": 0,
                 "verify_images": False},
        "model": {"name": "convnext_tiny",
                  "params": {"use_age": False, "stochastic_depth_prob": 0.1}},
        "train": {"epochs": 2, "warmup_epochs": 0, "batch_size": 4, "accumulation_steps": 2,
                  "learning_rate": 3e-4, "amp": True, "ema_decay": 0.9,
                  "ema_warmup_steps": 2, "mixup_alpha": 0.2, "cutmix_alpha": 1.0,
                  "label_smoothing": 0.05}})
    run = train(config)
    payload = read_checkpoint(run / "last.pt", "training")
    print("global_step", payload["global_step"], "ema_updates", payload["ema"]["num_updates"])
    assert payload["ema"]["num_updates"] == payload["global_step"] > 0
    difference = max((payload["ema"]["shadow"][key] - payload["model_state"][key]).abs().max().item()
                     for key in payload["ema"]["shadow"])
    print("max|ema-live|", difference)
    assert difference > 0
    model, loaded, age = load_model(run / "best.pt", torch.device("cpu"))
    print("best epoch", payload["best"]["epoch"], "best auroc", payload["best"]["metrics"]["auroc"])
    print("peak allocated MiB", round(torch.cuda.max_memory_allocated() / 2**20, 1))
