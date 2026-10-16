import math

import torch
from torch.nn import functional as F


def build_optimizer(model, config):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (no_decay if parameter.ndim <= 1 or name.endswith(".bias") else decay).append(parameter)
    return torch.optim.AdamW([{"params": decay, "weight_decay": config.weight_decay},
                              {"params": no_decay, "weight_decay": 0.0}], lr=config.learning_rate)


def learning_rate(step, total_steps, warmup_steps, config):
    if warmup_steps and step < warmup_steps:
        return config.learning_rate * (step + 1) / warmup_steps
    progress = min(1.0, max(0.0, (step - warmup_steps) / max(1, total_steps - warmup_steps - 1)))
    return config.min_learning_rate + (config.learning_rate - config.min_learning_rate) * (1 + math.cos(math.pi * progress)) / 2


def train_one_epoch(model, loader, optimizer, scaler, device, config, global_step, total_steps, warmup_steps):
    """Normalize gradients by samples in each window, including a short final window."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    amp = config.amp and device.type == "cuda"
    pos_weight = torch.tensor(config.pos_weight, device=device)
    loss_total, seen, skipped = 0.0, 0, 0
    for index, batch in enumerate(loader):
        if index % config.accumulation_steps == 0:
            remaining = len(loader.dataset) - seen
            window_samples = min(config.accumulation_steps * loader.batch_size, remaining)
        images = batch["image"].to(device, non_blocking=True)
        ages = batch["age"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = model(images, ages)
            if logits.shape != labels.shape:
                raise ValueError(f"Model must return [N] logits, got {tuple(logits.shape)}")
            loss_sum = F.binary_cross_entropy_with_logits(logits.float(), labels, pos_weight=pos_weight, reduction="sum")
        if not torch.isfinite(loss_sum):
            raise FloatingPointError("Non-finite training loss")
        scaler.scale(loss_sum / window_samples).backward()
        loss_total += loss_sum.detach().item()
        seen += len(labels)
        if (index + 1) % config.accumulation_steps == 0 or index + 1 == len(loader):
            rate = learning_rate(global_step, total_steps, warmup_steps, config)
            for group in optimizer.param_groups:
                group["lr"] = rate
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= previous_scale:
                global_step += 1
            else:
                skipped += 1
            optimizer.zero_grad(set_to_none=True)
        if (index + 1) % 100 == 0:
            print(f"[train] batch={index + 1}/{len(loader)} loss={loss_total / seen:.6f}", flush=True)
    return {"loss": loss_total / seen, "lr": optimizer.param_groups[0]["lr"],
            "skipped_optimizer_steps": skipped}, global_step
