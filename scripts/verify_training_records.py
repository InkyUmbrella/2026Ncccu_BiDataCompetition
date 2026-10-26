"""Verify the collected historical records; optionally verify downloaded weights."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=Path("training_records/20261022-23"))
    parser.add_argument("--weights-dir", type=Path, help="Extracted weights/ directory; omitted means binary weights are not checked")
    args = parser.parse_args()
    root = args.records.resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))["files"]
    for name, expected in manifest.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or digest(path) != expected:
            raise ValueError(f"Missing or changed historical record: {name}")
    checked = 0
    if args.weights_dir:
        weights = args.weights_dir.resolve()
        objects = json.loads((root / "retained_checkpoint_objects.json").read_text(encoding="utf-8"))
        for expected, row in objects.items():
            path = weights / Path(row["file"]).name
            if not path.is_file() or digest(path) != expected:
                raise ValueError(f"Missing or changed weight file: {path}")
            checked += 1
    print(json.dumps({"record_files_verified": len(manifest), "weight_files_verified": checked,
                      "weights_checked": args.weights_dir is not None,
                      "historical_per_epoch_archive_complete": False}, indent=2))


if __name__ == "__main__":
    main()
