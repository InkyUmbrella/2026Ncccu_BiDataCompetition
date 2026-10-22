"""Self-describing SSL encoder artifacts and resumable recovery checkpoints."""

from pathlib import Path

import torch

from ..utils import atomic_write
from .model import ENCODER_NAME

FORMAT_VERSION = 1


def unwrap(model):
    return model.module if hasattr(model, "module") else model


def encoder_payload(model, config, metadata, epoch, metrics):
    """Deployment/init artifact: encoder weights only, plain keys.

    The projector, momentum branches and queue are intentionally absent; the
    supervised trainer loads this into ``ConvNeXtClassifier.encoder``.
    """
    core = unwrap(model)
    payload = {
        "format_version": FORMAT_VERSION,
        "kind": "ssl_encoder",
        "model": {"name": ENCODER_NAME,
                  "params": {"stochastic_depth_prob": float(getattr(core, "stochastic_depth_prob", 0.0))}},
        "encoder_state": {key: value.detach().cpu().clone()
                          for key, value in core.encoder.state_dict().items()},
        "ssl_config": config.to_dict(),
        "data": metadata.get("data", {}),
        "augmentation": metadata.get("augmentation"),
        "epoch": int(epoch),
        "metrics": metrics,
        "environment": metadata.get("environment", {}),
    }
    return payload


def save_encoder(path, payload):
    atomic_write(path, lambda handle: torch.save(payload, handle))


def read_encoder(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != FORMAT_VERSION or payload.get("kind") != "ssl_encoder":
        raise ValueError(f"Expected version {FORMAT_VERSION} ssl_encoder checkpoint: {path}")
    return payload


def recovery_payload(model, optimizer, scaler, config, epoch, history, metrics, rng, device):
    """Everything needed to resume exactly (within RNG limits) on CPU/GPU."""
    core = unwrap(model)
    return {
        "format_version": FORMAT_VERSION,
        "kind": "ssl_training",
        "config": config.to_dict(),
        "epoch": int(epoch),
        "online_encoder": {k: v.detach().cpu().clone() for k, v in core.encoder.state_dict().items()},
        "online_projector": {k: v.detach().cpu().clone() for k, v in core.projector.state_dict().items()},
        "momentum_encoder": {k: v.detach().cpu().clone() for k, v in core.momentum_encoder.state_dict().items()},
        "momentum_projector": {k: v.detach().cpu().clone() for k, v in core.momentum_projector.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "queue": core.queue.detach().cpu().clone(),
        "queue_ptr": core.queue_ptr.detach().cpu().clone(),
        "queue_enqueued": int(core._enqueued.item()),
        "history": history,
        "metrics": metrics,
        "rng": rng,
    }


def save_recovery(path, payload):
    atomic_write(path, lambda handle: torch.save(payload, handle))


def read_recovery(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != FORMAT_VERSION or payload.get("kind") != "ssl_training":
        raise ValueError(f"Expected version {FORMAT_VERSION} ssl_training checkpoint: {path}")
    return payload


def load_recovery(model, optimizer, scaler, payload):
    core = unwrap(model)
    core.encoder.load_state_dict(payload["online_encoder"], strict=True)
    core.projector.load_state_dict(payload["online_projector"], strict=True)
    core.momentum_encoder.load_state_dict(payload["momentum_encoder"], strict=True)
    core.momentum_projector.load_state_dict(payload["momentum_projector"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    if "scaler" in payload:
        scaler.load_state_dict(payload["scaler"])
    core.queue.copy_(payload["queue"])
    core.queue_ptr.copy_(payload["queue_ptr"])
    core._enqueued[0] = int(payload.get("queue_enqueued", 0))


def sha256_or_none(path):
    if not path:
        return None
    from ..utils import sha256_file
    return sha256_file(Path(path))
