"""Explicitly download one allowed backbone plus pinned source/license evidence.

Standard library only. This script does not construct a model or start training.
Set TORCH_HOME for the later training process to this cache's parent structure.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import urllib.request
import uuid


REVISION = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"
DOCUMENTS = {
    "LICENSE": "600cc67cc4cb2f5ea317dcfc687ad1c74dc4bec8782bbe9db0afd83513b935b7",
    "MODEL_CARD.md": "70ca59606bee0a5fbb1baec80e7e29a93cd7cfbe26ca1910c52a852c4aab09d0",
    "README.md": "d1bc2e9686522bbd66ed6123dc81eb36b3a98e8ea9a2ca97778f54a1c641c9e1",
}


def download(url, destination, expected=None):
    digest = hashlib.sha256()
    request = urllib.request.Request(url, headers={"User-Agent": "glaucoma-source-audit/1"})
    with urllib.request.urlopen(request, timeout=60) as source, destination.open("xb") as output:
        while block := source.read(1024 * 1024):
            output.write(block)
            digest.update(block)
    value = digest.hexdigest()
    if expected and value != expected:
        raise ValueError(f"Source SHA256 mismatch: {destination}; retained for diagnosis")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True, help="TORCH_HOME/hub/checkpoints")
    parser.add_argument("--expected-sha256", help="Known full weight SHA256, if previously archived")
    args = parser.parse_args(argv)
    if args.expected_sha256 and not re.fullmatch("[0-9a-f]{64}", args.expected_sha256):
        raise ValueError("expected-sha256 must contain 64 lowercase hex characters")
    directory = Path(args.cache_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    names = ("dinov2_vits14_pretrain.pth", "dinov2_vits14_source.json", "dinov2_vits14_documents")
    if any((directory / name).exists() for name in names):
        raise FileExistsError("DINOv2 cache already contains artifacts; retain them and use a new cache if necessary")
    lock = directory / ".dinov2_vits14.prepare.lock"
    with lock.open("x", encoding="utf-8") as handle:
        handle.write("Exclusive preparation; do not run concurrent downloads.\n")
    try:
        staging = directory / ("dinov2_prepare_" + uuid.uuid4().hex)
        staging.mkdir()
        documents = staging / names[2]
        documents.mkdir()
        source_documents = {}
        for name, expected in DOCUMENTS.items():
            url = f"https://raw.githubusercontent.com/facebookresearch/dinov2/{REVISION}/{name}"
            digest = download(url, documents / name, expected)
            source_documents[name] = {"url": url, "sha256": digest}
        path = staging / names[0]
        digest = download(URL, path, args.expected_sha256)
        record = {"format_version": 1, "model": "dinov2_vits14", "weights_id": "LVD142M",
                  "url": URL, "sha256": digest, "bytes": path.stat().st_size,
                  "license": "Apache-2.0", "upstream_data": "LVD-142M (general-purpose images)",
                  "upstream_revision": REVISION, "documents": source_documents,
                  "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
                  "expected_weight_hash_supplied": args.expected_sha256 is not None}
        manifest = staging / names[1]
        manifest.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        for name in names:
            if (directory / name).exists():
                raise FileExistsError(f"Cache changed during preparation: {directory / name}")
        for name in names:
            (staging / name).rename(directory / name)
        print(json.dumps(record, indent=2))
    finally:
        lock.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
