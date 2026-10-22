import random

import numpy as np
import torch

from ..config import Config
from ..data.age import AgeStats
from ..models import build_model
from ..utils import atomic_write


def cpu_state(model):
    # Clone is also necessary on CPU: later optimizer steps must not mutate best weights.
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def inference_payload(model, config, age_stats, epoch, metrics):
    return {"format_version": 1, "kind": "inference", "config": config.to_dict(),
            "age_stats": age_stats.to_dict(), "model_state": cpu_state(model),
            "epoch": epoch, "metrics": metrics}


def save_checkpoint(path, payload):
    atomic_write(path, lambda handle: torch.save(payload, handle))


def read_checkpoint(path, kind):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1 or payload.get("kind") != kind:
        raise ValueError(f"Expected version 1 {kind} checkpoint: {path}")
    return payload


def load_model(path, device):
    payload = read_checkpoint(path, "inference")
    config = Config.from_dict(payload["config"])
    model = build_model(config.model)
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device).eval()
    return model, config, AgeStats(**payload["age_stats"])


def read_ssl_encoder(path):
    """Read a self-describing MoCo-v2 encoder artifact (kind='ssl_encoder')."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1 or payload.get("kind") != "ssl_encoder":
        raise ValueError(f"Expected version 1 ssl_encoder checkpoint: {path}")
    if "encoder_state" not in payload or "model" not in payload:
        raise ValueError(f"Malformed ssl_encoder checkpoint: {path}")
    return payload


def initialize_from_encoder(model, payload):
    """Load only the image encoder; the classifier/age modules stay fresh."""
    name = payload.get("model", {}).get("name")
    if name != "convnext_tiny":
        raise ValueError("Only convnext_tiny SSL encoders are supported")
    if not hasattr(model, "encoder"):
        raise ValueError("Target model has no image encoder")
    try:
        model.encoder.load_state_dict(payload["encoder_state"], strict=True)
    except RuntimeError as exc:
        raise ValueError(f"SSL encoder does not match the target encoder: {exc}") from exc


def initialize_model(model, config, payload):
    """Initialize a new experiment, without restoring optimizer/history/RNG."""
    source = Config.from_dict(payload["config"])
    if source.model == config.model:
        model.load_state_dict(payload["model_state"], strict=True)
    elif (source.model.name == "convnext_tiny"
          and not source.model.params.get("use_age", False)
          and config.model.name == "convnext_tiny_age_residual"):
        result = model.load_state_dict(payload["model_state"], strict=False)
        expected = {key for key in model.state_dict() if key.startswith("age_residual.")}
        if result.unexpected_keys or set(result.missing_keys) != expected:
            raise ValueError("B0 checkpoint does not match the residual image branch")
    else:
        raise ValueError("Initialization requires matching models or a B0 image checkpoint for age residual")


def rng_state(device):
    state = np.random.get_state()
    return {"python": random.getstate(), "numpy": [state[0], state[1].tolist(), state[2], state[3], state[4]],
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}


def restore_rng(state, device):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        if device.type != "cuda":
            raise ValueError("CUDA RNG state requires a CUDA device")
        torch.cuda.set_rng_state(state["cuda"], device)
