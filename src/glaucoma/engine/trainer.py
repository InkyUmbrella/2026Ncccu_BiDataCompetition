import math
from pathlib import Path
import time

import torch
from torch.optim.swa_utils import AveragedModel, get_ema_avg_fn
from torch.utils.data import DataLoader, WeightedRandomSampler

from ..config import Config
from ..data.age import AgeStats
from ..data.audit import audit_samples
from ..data.dataset import FundusDataset, read_samples
from ..data.transforms import build_transform
from ..models import build_model
from ..utils import (environment_metadata, new_run_directory, resolve_device,
                     save_json, seed_everything, seed_worker, sha256_file)
from .checkpoint import (cpu_state, inference_payload, initialize_model, read_checkpoint, restore_rng,
                         rng_state, save_checkpoint)
from .audit import finalize_run_audit, record_initialization, save_epoch_checkpoint, snapshot_sources
from .evaluation import collect_predictions, write_predictions
from .optimization import build_optimizer, train_one_epoch


def make_loader(dataset, batch_size, num_workers, seed, training=False, pin_memory=False,
                positive_fraction=None):
    sampler = None
    if positive_fraction is not None:
        if not training or not math.isfinite(positive_fraction) or not 0 < positive_fraction < 1:
            raise ValueError("positive_fraction requires training and a value in (0, 1)")
        labels = [sample.label for sample in dataset.samples]
        counts = {label: labels.count(label) for label in (0, 1)}
        if not all(counts.values()) or set(labels) != {0, 1}:
            raise ValueError("Positive sampling requires both binary classes")
        weights = [(positive_fraction if label else 1 - positive_fraction) / counts[label] for label in labels]
        sampler = WeightedRandomSampler(weights, len(dataset), replacement=True,
                                        generator=torch.Generator().manual_seed(seed))
    return DataLoader(dataset, batch_size=batch_size, shuffle=training and sampler is None,
                      sampler=sampler, drop_last=False,
                      num_workers=num_workers, pin_memory=pin_memory,
                      worker_init_fn=seed_worker,
                      generator=torch.Generator().manual_seed(seed), persistent_workers=False)


def _resume_compatible(config, saved):
    current, previous = config.to_dict(), Config.from_dict(saved).to_dict()
    # Relocation is supported; all numerical/data-loader settings remain fixed.
    for values in (current, previous):
        values.pop("output_root")
        values["data"].pop("root")
    if current != previous:
        raise ValueError("Resume config differs from checkpoint (only data.root/output_root may change)")


def train(config, resume=None, stop_after_epochs=None, image_cache=None, init_from=None):
    config.validate()
    if resume and init_from:
        raise ValueError("resume and init_from are mutually exclusive")
    if config.train.freeze_image_epochs and not (resume or init_from):
        raise ValueError("Freezing the image branch requires init_from or resume")
    if stop_after_epochs is not None and stop_after_epochs < 1:
        raise ValueError("stop_after_epochs must be positive")
    device = resolve_device(config.device)
    seed_everything(config.seed)
    saved = read_checkpoint(resume, "training") if resume else None
    initialization = read_checkpoint(init_from, "inference") if init_from else None
    start_epoch = 0
    if saved:
        _resume_compatible(config, saved["config"])
        start_epoch = saved["epoch"] + 1
        if start_epoch >= config.train.epochs:
            raise ValueError("This run has already completed all configured epochs")
    samples = {split: read_samples(config.data.root, split) for split in ("train", "valid")}
    for split, rows in samples.items():
        if {row.label for row in rows} != {0, 1}:
            raise ValueError(f"{split} must contain both labels for meaningful AUROC selection")
    csv_hashes = {split: sha256_file(Path(config.data.root) / f"{split}.csv") for split in samples}
    if saved and saved["csv_hashes"] != csv_hashes:
        raise ValueError("Dataset CSV contents changed since checkpoint")
    source = saved or initialization
    age_stats = AgeStats(**source["age_stats"]) if source else AgeStats.fit(row.age for row in samples["train"])
    run_dir = new_run_directory(config.output_root, config.experiment, config.seed)
    print(f"[run] {run_dir.resolve()}", flush=True)
    save_json(run_dir / "config.json", config.to_dict())
    save_json(run_dir / "age_stats.json", age_stats.to_dict())
    metadata = environment_metadata()
    metadata.update(csv_hashes=csv_hashes, resumed_from=str(Path(resume).resolve()) if resume else None,
                    initialized_from=str(Path(init_from).resolve()) if init_from else None,
                    image_cache=str(Path(image_cache).resolve()) if image_cache else None,
                    device=str(device), gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None)
    save_json(run_dir / "metadata.json", metadata)
    snapshot_sources(run_dir)
    caches = {}
    preprocessing = {"crop_black_border": config.data.crop_black_border,
                     "crop_threshold": config.data.crop_threshold, "crop_margin": config.data.crop_margin}
    if image_cache:
        from ..data.cache import ImageCache
        caches = {split: ImageCache(image_cache, rows, config.data.image_size, **preprocessing)
                  for split, rows in samples.items()}
    if config.data.verify_images:
        report = (audit_samples(samples, image_loader=lambda path: caches[path.parent.name].load(path))
                  if caches else audit_samples(samples))
        report["image_source"] = "lossless_cache" if caches else "original"
        save_json(run_dir / "data_audit.json", report)
        if report["errors"]:
            raise ValueError(f"Image decoding failed; see {run_dir / 'data_audit.json'}")
    datasets = {split: FundusDataset(rows, age_stats, build_transform(config.data.image_size, split == "train",
                                   preprocessed=split in caches, normalization=config.data.normalization,
                                   **preprocessing),
                                   image_cache=caches.get(split))
                for split, rows in samples.items()}
    valid_loader = make_loader(datasets["valid"], config.train.batch_size, config.data.num_workers,
                               config.seed, pin_memory=device.type == "cuda")
    batches = math.ceil(len(datasets["train"]) / config.train.batch_size)
    updates = math.ceil(batches / config.train.accumulation_steps)
    total_steps = updates * config.train.epochs
    warmup_steps = updates * config.train.warmup_epochs
    model = build_model(config.model, load_pretrained=source is None).to(device)
    if initialization:
        initialize_model(model, config, initialization)
    optimizer = build_optimizer(model, config.train)
    scaler = torch.amp.GradScaler("cuda", enabled=config.train.amp and device.type == "cuda")
    history, global_step, best = [], 0, None
    if saved:
        model.load_state_dict(saved["model_state"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        history, global_step, best = saved["history"], saved["global_step"], saved["best"]
        restore_rng(saved["rng"], device)
        save_checkpoint(run_dir / "best.pt", best)
    model_ema = None
    if config.train.ema_decay is not None:
        model_ema = AveragedModel(model, avg_fn=get_ema_avg_fn(config.train.ema_decay),
                                  use_buffers=True).requires_grad_(False)
        if saved:
            if "ema" not in saved:
                raise ValueError("EMA resume requires the saved EMA state")
            model_ema.load_state_dict(saved["ema"], strict=True)
    record_initialization(run_dir, model, config, age_stats, start_epoch,
                          source_file=resume or init_from,
                          mode="resume" if saved else "init_from" if initialization else "new",
                          parent_run=saved.get("run_directory") if saved else None)
    end_epoch = config.train.epochs if stop_after_epochs is None else min(config.train.epochs, start_epoch + stop_after_epochs)
    for epoch in range(start_epoch, end_epoch):
        started = time.monotonic()
        if hasattr(model, "set_image_frozen"):
            model.set_image_frozen(epoch < config.train.freeze_image_epochs)
        loader = make_loader(datasets["train"], config.train.batch_size, config.data.num_workers,
                             config.seed + epoch, training=True, pin_memory=device.type == "cuda",
                             positive_fraction=config.train.positive_fraction)
        training, global_step = train_one_epoch(model, loader, optimizer, scaler, device, config.train,
                                                global_step, total_steps, warmup_steps, model_ema)
        _, validation = collect_predictions(model, valid_loader, device, config.train.pos_weight, config.train.amp)
        selected_model, variant = model, "raw"
        raw_validation = validation
        ema_validation = None
        if model_ema is not None:
            _, ema_validation = collect_predictions(model_ema.module, valid_loader, device,
                                                    config.train.pos_weight, config.train.amp)
            if ((ema_validation["auroc"], ema_validation["auprc_trapz"])
                    > (validation["auroc"], validation["auprc_trapz"])):
                selected_model, variant, validation = model_ema.module, "ema", ema_validation
        row = {"epoch": epoch, "global_step": global_step, "train": training, "valid": validation,
               "image_frozen": bool(getattr(model, "image_frozen", False)),
               "seconds": round(time.monotonic() - started, 3)}
        if model_ema is not None:
            row.update(valid_raw=raw_validation, valid_ema=ema_validation, selected_weights=variant)
        history.append(row)
        score = (validation["auroc"], validation["auprc_trapz"])
        if best is None or score > (best["metrics"]["auroc"], best["metrics"]["auprc_trapz"]):
            best = inference_payload(selected_model, config, age_stats, epoch, validation)
            if model_ema is not None:
                best["weights_variant"] = variant
            save_checkpoint(run_dir / "best.pt", best)
        # A self-contained recovery file includes the best weights as well as the
        # current optimizer state; it is not part of the deployment artifact.
        save_epoch_checkpoint(run_dir, {
            "format_version": 1, "kind": "training", "config": config.to_dict(),
            "run_directory": str(run_dir.resolve()),
            "age_stats": age_stats.to_dict(), "csv_hashes": csv_hashes,
            "epoch": epoch, "global_step": global_step, "model_state": cpu_state(model),
            "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
            "rng": rng_state(device), "history": history, "best": best,
            **({"ema": cpu_state(model_ema)} if model_ema is not None else {})})
        save_json(run_dir / "history.json", history)
        print(f"[epoch {epoch + 1}/{config.train.epochs}] train_loss={training['loss']:.6f} "
              f"valid_auc={validation['auroc']:.6f} valid_ap={validation['ap']:.6f}", flush=True)
    model.load_state_dict(best["model_state"])
    predictions, metrics = collect_predictions(model, valid_loader, device, config.train.pos_weight, config.train.amp)
    write_predictions(run_dir / "valid_predictions.csv", predictions)
    save_json(run_dir / "metrics.json", {"best_epoch": best["epoch"], "completed_epochs": end_epoch,
                                         "training_complete": end_epoch == config.train.epochs, "valid": metrics,
                                         **({"best_weights_variant": best["weights_variant"]}
                                            if model_ema is not None else {})})
    finalize_run_audit(run_dir)
    return run_dir
