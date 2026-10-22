"""MoCo-v2 pretraining loop on the provided training split only.

The supervised trainer in :mod:`glaucoma.engine.trainer` is deliberately not
modified by this module; ``--init-encoder`` is the only bridge.
"""

import math
import os
from pathlib import Path
import random
import time

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ..data.age import AgeStats
from ..data.dataset import FundusDataset, read_samples
from ..utils import (environment_metadata, new_run_directory, resolve_device,
                     save_json, seed_everything, seed_worker, sha256_file)
from .augment import AUGMENTATION_SPEC, build_probe_transform, build_ssl_transform
from .checkpoint import (encoder_payload, load_recovery, read_recovery, recovery_payload,
                         save_encoder, save_recovery, unwrap)
from .config import SSLConfig
from .model import MoCoV2
from .probe import effective_rank, extract_features, feature_std, probe_metrics


def ssl_learning_rate(step, total_steps, warmup_steps, base_lr):
    """Linear warmup then cosine decay to zero (per optimizer step)."""
    if total_steps < 1:
        raise ValueError("total_steps must be positive")
    if warmup_steps and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = min(1.0, max(0.0, (step - warmup_steps) / max(1, total_steps - warmup_steps)))
    return base_lr * (1 + math.cos(math.pi * progress)) / 2


def distributed_info():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("Invalid torchrun environment (WORLD_SIZE/RANK)")
    return world_size, rank, local_rank


def _preprocessing(config):
    return {"crop_black_border": config.data.crop_black_border,
            "crop_threshold": config.data.crop_threshold,
            "crop_margin": config.data.crop_margin,
            "decode": config.data.decode}


def load_train_rows(config):
    """Read the train split unlabelled and hard-fail on any other split."""
    rows = read_samples(config.data.root, "train", require_labels=False)
    if not rows:
        raise ValueError("Empty train split")
    root = Path(config.data.root).resolve()
    for row in rows:
        if row.path.parent.name != "train":
            raise ValueError(f"SSL may only read the train split, got {row.path}")
        if row.path.parent.parent.resolve() != root:
            raise ValueError(f"SSL row escapes the configured data root: {row.path}")
    return rows


def make_ssl_loader(dataset, batch_size, num_workers, seed, epoch, *, world_size=1,
                    rank=0, sampler=None, pin_memory=False):
    if sampler is not None:
        sampler.set_epoch(epoch)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=True,
                          num_workers=num_workers, pin_memory=pin_memory,
                          worker_init_fn=seed_worker, persistent_workers=num_workers > 0)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                      num_workers=num_workers, pin_memory=pin_memory,
                      worker_init_fn=seed_worker,
                      generator=torch.Generator().manual_seed(seed + epoch),
                      persistent_workers=num_workers > 0)


def build_optimizer(model, config):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (no_decay if parameter.ndim <= 1 or name.endswith(".bias") else decay).append(parameter)
    groups = [{"params": decay, "weight_decay": config.train.weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.SGD(groups, lr=config.train.learning_rate,
                           momentum=config.train.momentum)


def _train_probe(config, cache, encoder, device, amp):
    """Fit/evaluate a frozen linear probe on disjoint subsets of **train** only."""
    if cache is None:
        return {"skipped": "probe requires the lossless cache"}
    labeled = read_samples(config.data.root, "train", require_labels=True)
    if any(row.path.parent.name != "train" for row in labeled):
        raise ValueError("The SSL probe may only read the train split")
    take = min(len(labeled), config.train.probe_fit_size + config.train.probe_eval_size)
    if take < 4:
        return {"skipped": "not enough train rows"}
    rng = random.Random(config.seed)
    indices = sorted(rng.sample(range(len(labeled)), take))
    fit_count = max(1, min(config.train.probe_fit_size, take - 1))
    groups = {"fit": [labeled[i] for i in indices[:fit_count]],
              "eval": [labeled[i] for i in indices[fit_count:]]}
    for name, rows in groups.items():
        if {row.label for row in rows} != {0, 1}:
            return {"skipped": f"{name} subset lacks both classes"}
    age_stats = AgeStats.fit(row.age for row in labeled)
    transform = build_probe_transform(config.data.image_size,
                                      crop_black_border=config.data.crop_black_border,
                                      crop_threshold=config.data.crop_threshold,
                                      crop_margin=config.data.crop_margin,
                                      preprocessed=True)
    features = {}
    for name, rows in groups.items():
        dataset = FundusDataset(rows, age_stats, transform, image_cache=cache,
                                image_size=config.data.image_size)
        loader = DataLoader(dataset, batch_size=min(128, config.train.batch_size),
                            shuffle=False, drop_last=False, num_workers=config.data.num_workers,
                            pin_memory=device.type == "cuda")
        x, y, _ = extract_features(encoder, loader, device, amp)
        features[name] = (x, y)
    result = probe_metrics(*features["fit"], *features["eval"])
    result.update(fit_size=len(groups["fit"]), eval_size=len(groups["eval"]),
                  eval_positive=int(features["eval"][1].sum()),
                  effective_rank=effective_rank(features["eval"][0]),
                  feature_std=feature_std(features["eval"][0]))
    return result


def _resume_compatible(config, saved):
    current, previous = config.to_dict(), SSLConfig.from_dict(saved).to_dict()
    for values in (current, previous):
        values.pop("output_root")
        values["data"].pop("root")
        values["data"].pop("cache")
    if current != previous:
        raise ValueError("Resume config differs from the SSL checkpoint (only data root/cache/output_root may change)")


def pretrain(config, resume=None, stop_after_epochs=None, image_cache=None, device=None):
    config.validate()
    if stop_after_epochs is not None and stop_after_epochs < 1:
        raise ValueError("stop_after_epochs must be positive")
    world_size, rank, local_rank = distributed_info()
    is_main = rank == 0
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed SSL requires CUDA")
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = resolve_device(device or config.device)
    if config.train.batch_size % world_size:
        raise ValueError("batch_size must be divisible by the number of processes")
    seed_everything(config.seed + rank)
    amp = config.train.amp and device.type == "cuda"
    cache_dir = image_cache or config.data.cache

    rows = load_train_rows(config)
    age_stats = AgeStats.fit(row.age for row in rows)
    preprocessing = _preprocessing(config)
    cache = None
    if cache_dir:
        from ..data.cache import ImageCache
        cache = ImageCache(cache_dir, rows, config.data.image_size, **preprocessing)
    elif is_main:
        print("[ssl] warning: no image cache; decoding NFS JPEGs will be slow", flush=True)
    dataset = FundusDataset(rows, age_stats,
                            build_ssl_transform(config.data.ssl_size,
                                                min_crop_scale=config.data.min_crop_scale),
                            image_cache=cache, decode=config.data.decode,
                            image_size=config.data.image_size)

    run_dir = new_run_directory(config.output_root, config.experiment, config.seed)
    csv_sha = sha256_file(Path(config.data.root) / "train.csv")
    manifest_sha = (sha256_file(Path(cache_dir) / "manifest.json") if cache_dir else None)
    data_metadata = {"split": "train", "count": len(rows), "csv_sha256": csv_sha,
                     "cache": str(Path(cache_dir).resolve()) if cache_dir else None,
                     "cache_manifest_sha256": manifest_sha,
                     "cache_preprocessing": preprocessing,
                     "image_size": config.data.image_size, "ssl_size": config.data.ssl_size}
    metadata = environment_metadata()
    metadata.update(run_dir=str(run_dir.resolve()), world_size=world_size, rank=rank,
                    local_rank=local_rank, device=str(device),
                    gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                    data=data_metadata, augmentation=AUGMENTATION_SPEC,
                    batch_size_per_process=config.train.batch_size // world_size)
    if is_main:
        save_json(run_dir / "config.json", config.to_dict())
        save_json(run_dir / "metadata.json", metadata)
        print(f"[ssl] {run_dir.resolve()}  train_rows={len(rows)}  world_size={world_size}", flush=True)

    model = MoCoV2(queue_size=config.train.queue_size,
                   temperature=config.train.temperature,
                   momentum=config.train.moco_momentum,
                   stochastic_depth_prob=0.0).to(device)
    optimizer = build_optimizer(model, config)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    saved = read_recovery(resume) if resume else None
    start_epoch = 0
    history, probe_records, best_score = [], [], None
    if saved:
        _resume_compatible(config, saved["config"])
        start_epoch = saved["epoch"] + 1
        if start_epoch >= config.train.epochs:
            raise ValueError("This SSL run has already completed all configured epochs")
        load_recovery(model, optimizer, scaler, saved)
        history = saved["history"]
        probe_records = list(saved.get("metrics", {}).get("probes", []))
        from ..engine.checkpoint import restore_rng
        restore_rng(saved["rng"], device)

    if world_size > 1:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=False)
    core = unwrap(model)

    sampler = None
    if world_size > 1:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(dataset, num_workers=config.data.num_workers,
                                     shuffle=True, seed=config.seed, drop_last=True)
    steps_per_epoch = len(dataset) // (config.train.batch_size // world_size)
    total_steps = steps_per_epoch * config.train.epochs
    warmup_steps = steps_per_epoch * config.train.warmup_epochs
    end_epoch = config.train.epochs if stop_after_epochs is None else min(
        config.train.epochs, start_epoch + stop_after_epochs)
    global_step = start_epoch * steps_per_epoch

    for epoch in range(start_epoch, end_epoch):
        started = time.monotonic()
        if sampler is not None:
            sampler.set_epoch(epoch)
        loader = make_ssl_loader(dataset, config.train.batch_size // world_size,
                                 config.data.num_workers, config.seed, epoch,
                                 world_size=world_size, rank=rank, sampler=sampler,
                                 pin_memory=device.type == "cuda")
        model.train()
        loss_total, seen, skipped = 0.0, 0, 0
        last_keys = None
        for index, batch in enumerate(loader):
            views = batch["image"]
            if isinstance(views, torch.Tensor):
                if views.ndim != 5:
                    raise ValueError("Expected two stacked views from the loader")
                view_q, view_k = views[:, 0], views[:, 1]
            else:
                view_q, view_k = views[0], views[1]
            view_q = view_q.to(device, non_blocking=True)
            view_k = view_k.to(device, non_blocking=True)
            rate = ssl_learning_rate(global_step, total_steps, warmup_steps,
                                     config.train.learning_rate)
            for group in optimizer.param_groups:
                group["lr"] = rate
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits, targets, keys = model(view_q, view_k, return_keys=True)
            loss = F.cross_entropy(logits.float(), targets)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite SSL loss")
            scaler.scale(loss).backward()
            if config.train.grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= previous_scale:
                core.momentum_update()
                core.dequeue_and_enqueue(keys)
                global_step += 1
            else:
                skipped += 1
            loss_total += float(loss.detach()) * view_q.shape[0]
            seen += view_q.shape[0]
            last_keys = keys.detach()
            if is_main and (index + 1) % 100 == 0:
                print(f"[ssl] epoch={epoch + 1} batch={index + 1}/{steps_per_epoch} "
                      f"loss={loss_total / max(seen, 1):.4f}", flush=True)
        if not seen:
            raise ValueError("The train loader produced no samples")
        metrics = {"loss": loss_total / seen, "lr": optimizer.param_groups[0]["lr"],
                   "skipped_optimizer_steps": skipped,
                   "queue_fill": core.queue_fill if hasattr(core, "queue_fill") else 0}
        keys_cpu = last_keys.float().cpu().numpy() if last_keys is not None else None
        if keys_cpu is not None:
            metrics.update(effective_rank=effective_rank(keys_cpu),
                           feature_std=feature_std(keys_cpu))
        finished = epoch + 1
        if finished % config.train.probe_every == 0 or finished == config.train.epochs:
            if is_main:
                probe = _train_probe(config, cache, core.encoder, device, amp)
                probe["epoch"] = epoch
                probe_records.append(probe)
                score = probe.get("auroc")
                if score is not None and (best_score is None or score > best_score):
                    best_score = score
                    save_encoder(run_dir / "encoder_best.pt",
                                 encoder_payload(model, config, metadata, epoch, probe))
                metrics["probe"] = probe
        row = {"epoch": epoch, "global_step": global_step, "train": metrics,
               "seconds": round(time.monotonic() - started, 3)}
        history.append(row)
        if is_main:
            save_json(run_dir / "history.json", history)
            if probe_records:
                save_json(run_dir / "probe.json", probe_records)
            if (finished % config.train.save_every == 0 or finished == config.train.epochs
                    or finished == end_epoch):
                save_encoder(run_dir / "encoder_last.pt",
                             encoder_payload(model, config, metadata, epoch, metrics))
                recovery = recovery_payload(model, optimizer, scaler, config, epoch, history,
                                            {"probes": probe_records}, _rng(device), device)
                save_recovery(run_dir / "last.pt", recovery)
            print(f"[ssl] epoch {finished}/{config.train.epochs} loss={metrics['loss']:.4f} "
                  f"lr={metrics['lr']:.5f} queue={metrics['queue_fill']} "
                  f"rank={metrics.get('effective_rank', float('nan')):.2f} "
                  f"std={metrics.get('feature_std', float('nan')):.4f}"
                  + (f" probe_auc={metrics['probe'].get('auroc')}" if "probe" in metrics else ""),
                  flush=True)

    if world_size > 1:
        torch.distributed.barrier()
    if is_main:
        if not (run_dir / "encoder_last.pt").exists():
            save_encoder(run_dir / "encoder_last.pt",
                         encoder_payload(model, config, metadata, end_epoch - 1, history[-1]["train"]))
        if not (run_dir / "encoder_best.pt").exists():
            save_encoder(run_dir / "encoder_best.pt",
                         encoder_payload(model, config, metadata, end_epoch - 1,
                                         history[-1]["train"] if history else {}))
        artifacts = {}
        for name in ("config.json", "metadata.json", "history.json", "encoder_last.pt",
                     "encoder_best.pt", "last.pt", "probe.json"):
            path = run_dir / name
            if path.exists():
                artifacts[name] = sha256_file(path)
        save_json(run_dir / "manifest.json", {"artifacts": artifacts,
                                              "algorithm": "moco_v2",
                                              "augmentation": AUGMENTATION_SPEC})
    if world_size > 1 and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    return run_dir


def _rng(device):
    from ..engine.checkpoint import rng_state
    return rng_state(device)
