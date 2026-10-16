import csv
from pathlib import Path

import torch
from torch.nn import functional as F

from ..metrics import age_stratified_metrics, binary_metrics


@torch.inference_mode()
def collect_predictions(model, loader, device, pos_weight=5.0, amp=False):
    model.eval()
    result = {"names": [], "logits": [], "probabilities": [], "targets": [], "ages": []}
    loss_sum = 0.0
    for batch in loader:
        images, ages = batch["image"].to(device), batch["age"].to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            logits = model(images, ages).float()
        if logits.shape != (len(images),) or not torch.isfinite(logits).all():
            raise ValueError("Model must return finite logits of shape [N]")
        result["names"].extend(batch["name"])
        result["ages"].extend(batch["raw_age"].tolist())
        result["logits"].extend(logits.cpu().tolist())
        result["probabilities"].extend(logits.sigmoid().cpu().tolist())
        if "label" in batch:
            labels = batch["label"].to(device)
            loss_sum += F.binary_cross_entropy_with_logits(
                logits, labels, pos_weight=torch.tensor(pos_weight, device=device), reduction="sum").item()
            result["targets"].extend(labels.int().cpu().tolist())
    metrics = None
    if result["targets"]:
        metrics = binary_metrics(result["targets"], result["probabilities"])
        metrics["loss"] = loss_sum / len(result["targets"])
        metrics["age_groups"] = age_stratified_metrics(result["targets"], result["probabilities"], result["ages"])
    return result, metrics


def write_predictions(path, predictions, submission=False):
    """Exclusive creation avoids silently replacing an existing submission."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["Image_name", "Probability", "Pred_Label"] if submission else ["Image_name", "True_Label", "Logit", "Probability"]
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        for index, name in enumerate(predictions["names"]):
            probability = predictions["probabilities"][index]
            if submission:
                writer.writerow([name, format(probability, ".10g"), int(probability >= 0.5)])
            else:
                writer.writerow([name, predictions["targets"][index],
                                 format(predictions["logits"][index], ".10g"), format(probability, ".10g")])
