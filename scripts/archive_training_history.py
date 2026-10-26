"""Read-only audit of the four completed 2026-10-22/23 server experiments.

Run on the training server with CUDA_VISIBLE_DEVICES empty. Historical files are
never changed. This collects existing evidence; it cannot recreate lost epochs.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tarfile

import numpy as np
import torch
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def finite(value):
    if isinstance(value, torch.Tensor):
        assert torch.isfinite(value).all().item(), "Non-finite checkpoint tensor"
    elif isinstance(value, dict):
        for item in value.values():
            finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            finite(item)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-home", type=Path, default=Path("/home/lichunyu"))
    parser.add_argument("--output", type=Path, required=True, help="New independent archive directory")
    args = parser.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "This audit must run without a visible GPU"
    torch.set_num_threads(2)
    home, output = args.server_home.resolve(), args.output.resolve()
    repos = {"b0": home / "competition-pretrained", "age": home / "competition-fwq-pretrained",
             "improve": home / "competition-pretrained-improve"}
    launch = read(repos["improve"] / "outputs/pretrained-improve/launch.json")
    attempt = Path(launch["run_root"])
    status, context = read(attempt / "status.json"), read(attempt / "context.json")
    assert status["stage"] == "complete" and (attempt / "supervisor.exit").read_text().strip() == "0"
    transfer = repos["age"] / "outputs/fwq-transfer/transfer_20261022T134645Z"
    migration = repos["b0"] / "outputs/pretrained-validation/migrations/gpu2-to-gpu4_20261022T142431Z"
    runs = {
        "b0": repos["b0"] / "outputs/pretrained/b0_crop384_imagenet/seed42_20261022T142612Z_8827a159",
        "b0_before_migration": repos["b0"] / "outputs/pretrained/b0_crop384_imagenet/seed42_20261022T103749Z_393b6b2d",
        "age": transfer / "age_training/b0_crop384_age_residual_imagenet/seed42_20261022T135818Z_9aec5250",
        **{name: Path(status["jobs"][name]["run_directory"]) for name in ("refine", "restart")}}
    logs = {"b0": [migration / "previous_train.log", migration / "train_gpu4.log"],
            "age": [transfer / "age_training.log"],
            "refine": [attempt / "refine.log"], "restart": [attempt / "restart.log"]}
    exits = {"b0": migration / "train_gpu4.exit", "age": transfer / "age_training.exit",
             "refine": attempt / "refine.exit", "restart": attempt / "restart.exit"}
    initializers = {"b0": home / ".cache/torch/hub/checkpoints/convnext_tiny-983f1562.pth",
                    "age": transfer / "b0_snapshot.pt", "refine": Path(context["source_snapshot"])}
    initializers["restart"] = initializers["b0"]
    roots = [repos["b0"] / "outputs/pretrained", repos["b0"] / "outputs/pretrained-validation",
             repos["age"] / "outputs/fwq-transfer", repos["improve"] / "outputs/pretrained-improve"]
    assert not any(output == repo or output.is_relative_to(repo) for repo in repos.values())
    output.mkdir(parents=True, exist_ok=False)
    records = output / "records"
    records.mkdir()
    (output / "weights").mkdir()
    observed = datetime.now(timezone.utc).isoformat()
    inventory, objects, mapping = [], {}, {}

    def collect(path):
        path = path.resolve()
        stat = path.stat()
        digest = sha(path)
        binary = path.suffix in (".pt", ".pth")
        if binary:
            target = output / "weights" / (digest + path.suffix)
            if digest not in objects:
                shutil.copy2(path, target)
                assert sha(target) == digest
                payload = torch.load(target, map_location="cpu", weights_only=True)
                finite(payload)
                objects[digest] = {"file": target.relative_to(output).as_posix(), "bytes": stat.st_size,
                                   "kind": payload.get("kind", "official_pretrained_state_dict"),
                                   "epoch": int(payload["epoch"]) + 1 if "epoch" in payload else None}
                del payload
            archived = objects[digest]["file"]
        else:
            named_run = next((name for name, run in runs.items() if path.is_relative_to(run)), None)
            repo_name = next((name for name, repo in repos.items() if path.is_relative_to(repo)), "data")
            relative = (Path("runs") / named_run / path.relative_to(runs[named_run]) if named_run else
                        Path("support") / repo_name / f"{len(inventory):03d}_{path.name}")
            target = records / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            assert sha(target) == digest
            archived = "records/" + relative.as_posix()
        after = path.stat()
        assert (stat.st_size, stat.st_mtime_ns) == (after.st_size, after.st_mtime_ns), str(path)
        row = {"original_path": str(path), "archive_file": archived, "sha256": digest,
               "bytes": stat.st_size, "observed_mtime_ns": stat.st_mtime_ns,
               "observed_mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
               "collected_at_utc": observed}
        inventory.append(row)
        mapping[str(path)] = row

    print("Collecting original records and retained checkpoint files", flush=True)
    for root in roots:
        for path in sorted(root.rglob("*")):
            if path.is_file() and (path.suffix in {".json", ".csv", ".log", ".exit", ".sh", ".py", ".pt", ".pth"}
                                   or path.name.endswith(".tar.gz")):
                collect(path)
    collect(initializers["b0"])
    collect(Path(context["cache"]) / "manifest.json")

    provenance = {}
    for alias, repo in repos.items():
        package = repo / "src/glaucoma"
        aggregate = hashlib.sha256()
        files = {}
        paths = [p for folder in ("src", "configs", "scripts", "tests", "docs")
                 for p in (repo / folder).rglob("*") if p.is_file() and p.suffix in {".py", ".json", ".md", ".sh"}]
        paths += [repo / name for name in ("pyproject.toml", "README.md", ".gitignore") if (repo / name).is_file()]
        for path in sorted(package.rglob("*.py")):
            aggregate.update(path.relative_to(package).as_posix().encode())
            aggregate.update(path.read_bytes())
        destination = records / "sources" / (alias + ".tar.gz")
        destination.parent.mkdir(exist_ok=True)
        with tarfile.open(destination, "w:gz") as archive:
            for path in sorted(set(paths)):
                name = path.relative_to(repo).as_posix()
                files[name] = sha(path)
                archive.add(path, arcname=name)
        provenance[alias] = {"original_repository": str(repo), "archive": destination.relative_to(records).as_posix(),
                             "archive_sha256": sha(destination), "runtime_package_sha256": aggregate.hexdigest(),
                             "captured_at_utc": observed, "files": files}
    save(records / "source_provenance.json", provenance)

    data_root = Path(context["data_root"])
    csv_hashes = {split: sha(data_root / (split + ".csv")) for split in ("train", "valid")}
    assert csv_hashes == context["csv_hashes"]
    with (data_root / "valid.csv").open(newline="", encoding="utf-8-sig") as stream:
        official = list(csv.DictReader(stream))
    labels = np.array([int(row["True_Label"]) for row in official])
    assert len(labels) == 5849 and int(labels.sum()) == 191
    probabilities, results = {}, {}
    checkpoints = {row["original_path"]: row for row in inventory if row["archive_file"].startswith("weights/")}
    for name in ("b0", "age", "refine", "restart"):
        print("Verifying " + name, flush=True)
        run = runs[name]
        config, metadata, metrics, history = [read(run / (key + ".json")) for key in ("config", "metadata", "metrics", "history")]
        epochs = config["train"]["epochs"]
        assert metrics["training_complete"] and metrics["completed_epochs"] == epochs
        assert len(history) == epochs and [row["epoch"] for row in history] == list(range(epochs))
        assert config["seed"] == 42 and metadata["csv_hashes"] == csv_hashes
        repo_alias = name if name in ("b0", "age") else "improve"
        assert metadata["source_sha256"] == provenance[repo_alias]["runtime_package_sha256"]
        best, last = [torch.load(run / file, map_location="cpu", weights_only=True) for file in ("best.pt", "last.pt")]
        assert best["kind"] == "inference" and last["kind"] == "training"
        assert last["epoch"] + 1 == epochs and last["history"] == history
        assert last["optimizer"]["state"] and last["scaler"] and last["rng"]["cuda"] is not None
        assert best["epoch"] == metrics["best_epoch"] == last["best"]["epoch"]
        assert all(torch.equal(value, last["best"]["model_state"][key]) for key, value in best["model_state"].items())
        chosen = max(history, key=lambda row: (row["valid"]["auroc"], row["valid"]["auprc_trapz"]))
        assert chosen["epoch"] == best["epoch"]
        if name in ("refine", "restart"):
            assert int(last["ema"]["n_averaged"]) == last["global_step"]
        with (run / "valid_predictions.csv").open(newline="", encoding="utf-8") as stream:
            predictions = list(csv.DictReader(stream))
        assert [row["Image_name"] for row in predictions] == [row["Image_name"] for row in official]
        assert [int(row["True_Label"]) for row in predictions] == labels.tolist()
        scores = np.array([float(row["Probability"]) for row in predictions])
        assert np.isfinite(scores).all() and ((0 <= scores) & (scores <= 1)).all()
        precision, recall, _ = precision_recall_curve(labels, scores)
        recomputed = {"auroc": float(roc_auc_score(labels, scores)), "ap": float(average_precision_score(labels, scores)),
                      "auprc_trapz": float(auc(recall, precision))}
        assert all(abs(recomputed[key] - metrics["valid"][key]) < 1e-10 for key in recomputed)
        observed_epochs = sorted({int(epoch) for log in logs[name]
                                  for epoch in re.findall(r"^\[epoch (\d+)/\d+\]", log.read_text(), flags=re.M)})
        assert observed_epochs == list(range(1, epochs + 1)), (name, observed_epochs)
        assert exits[name].read_text().strip() == "0"
        initialization = checkpoints[str(initializers[name])]
        available = sorted({objects[row["sha256"]]["epoch"] for path, row in checkpoints.items()
                            if path.startswith(str(repos["b0"] if name == "b0" else run) + "/")
                            and objects[row["sha256"]]["kind"] == "training"})
        results[name] = {"run_directory": str(run), "completed_epochs": epochs, "training_exit_code": 0,
                         "best_epoch": best["epoch"] + 1, "weights_variant": best.get("weights_variant", "raw"),
                         "metrics": recomputed, "seed": config["seed"], "global_step": last["global_step"],
                         "initialization": initialization, "best": checkpoints[str(run / "best.pt")],
                         "last": checkpoints[str(run / "last.pt")], "logs": [mapping[str(log)] for log in logs[name]],
                         "predictions": mapping[str(run / "valid_predictions.csv")],
                         "source_package_matches_training_metadata": True, "source_snapshot": provenance[repo_alias]["archive"],
                         "available_full_training_checkpoint_epochs": available,
                         "missing_full_training_checkpoint_epochs": sorted(set(range(1, epochs + 1)) - set(available)),
                         "per_epoch_archive_complete": False, "fault_recoveries": 0,
                         "user_requested_device_migrations": 1 if name == "b0" else 0}
        if name in ("refine", "restart"):
            for variant in ("raw", "ema"):
                row = max(history, key=lambda row: (row["valid_" + variant]["auroc"], row["valid_" + variant]["auprc_trapz"]))
                results[name]["best_" + variant] = {"epoch": row["epoch"] + 1, "metrics": row["valid_" + variant]}
            results[name]["ema_n_averaged"] = int(last["ema"]["n_averaged"])
        probabilities[name] = scores
        del best, last

    print("Recomputing paired stratified bootstrap intervals", flush=True)
    positive, negative = np.flatnonzero(labels), np.flatnonzero(1 - labels)
    for name, result in results.items():
        result["delta_from_b0"] = {key: result["metrics"][key] - results["b0"]["metrics"][key] for key in result["metrics"]}
        rng, deltas = np.random.default_rng(20261023), []
        if name != "b0":
            for _ in range(1000):
                indices = np.concatenate((rng.choice(positive, len(positive)), rng.choice(negative, len(negative))))
                deltas.append(roc_auc_score(labels[indices], probabilities[name][indices]) -
                              roc_auc_score(labels[indices], probabilities["b0"][indices]))
        result["paired_bootstrap_auroc_delta_95ci"] = np.quantile(deltas, [0.025, 0.975]).tolist() if deltas else [0., 0.]
    summary = {"collected_at_utc": observed, "submission_status": "User previously confirmed not submitted; no platform access performed",
               "train_samples": 51396, "valid_samples": 5849, "valid_positives": 191, "csv_sha256": csv_hashes,
               "bootstrap_replicates": 1000, "bootstrap_seed": 20261023,
               "bootstrap_limitation": "Image-level paired stratified intervals do not remove repeated model-selection bias",
               "all_four_completed_and_artifacts_verified": True, "historical_per_epoch_archive_complete": False,
               "results": results}
    save(records / "results.json", summary)
    save(records / "original_file_inventory.json", inventory)
    save(records / "retained_checkpoint_objects.json", objects)
    save(records / "collection_scope.json", {"roots_searched": [str(root) for root in roots],
         "weights_total_bytes": sum(row["bytes"] for row in objects.values()),
         "timestamp_semantics": "Current observation of existing file mtime; no historical hash ledger was created retroactively",
         "missing_materials": ["Most epoch-specific checkpoint files were overwritten by the historical trainer",
                               "No contemporaneous per-epoch timestamp/SHA256 ledger",
                               "No original model_initial.pt snapshots (pretrained/init-from source files are retained)"],
         "not_collected": ["Raw dataset images", "Competition test predictions", "Unrelated projects or users"]})
    save(records / "manifest.json", {"files": {p.relative_to(records).as_posix(): sha(p) for p in sorted(records.rglob("*")) if p.is_file()}})
    with tarfile.open(output / "records.tar.gz", "w:gz") as archive:
        archive.add(records, arcname="records")
    # All retained binaries are also kept as one self-contained server backup.
    print("Creating binary archive", flush=True)
    with tarfile.open(output / "weights.tar", "w") as archive:
        archive.add(output / "weights", arcname="weights")
    save(output / "archive_manifest.json", {name: {"sha256": sha(output / name), "bytes": (output / name).stat().st_size}
                                          for name in ("records.tar.gz", "weights.tar")})
    (output / "collection.exit").write_text("0\n")
    print(json.dumps({"output": str(output), "records": len(inventory), "unique_weight_files": len(objects),
                      "results": {name: row["metrics"] for name, row in results.items()}}), flush=True)


if __name__ == "__main__":
    main()
