"""Offline, hash-checked source of the original general-purpose DINOv2 weights."""

import json
from pathlib import Path

import torch

from ..utils import sha256_file


DINOV2_REVISION = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
DINOV2_URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"
DINOV2_FILE = "dinov2_vits14_pretrain.pth"
DINOV2_DOCUMENTS = {
    "LICENSE": "600cc67cc4cb2f5ea317dcfc687ad1c74dc4bec8782bbe9db0afd83513b935b7",
    "MODEL_CARD.md": "70ca59606bee0a5fbb1baec80e7e29a93cd7cfbe26ca1910c52a852c4aab09d0",
    "README.md": "d1bc2e9686522bbd66ed6123dc81eb36b3a98e8ea9a2ca97778f54a1c641c9e1",
}


def dinov2_source():
    directory = (Path(torch.hub.get_dir()) / "checkpoints").resolve()
    path = directory / DINOV2_FILE
    manifest = directory / "dinov2_vits14_source.json"
    if not path.is_file() or not manifest.is_file():
        raise FileNotFoundError("Prepare original DINOv2 weights first with scripts/prepare_dinov2.py; "
                                f"expected {path} and {manifest}")
    record = json.loads(manifest.read_text(encoding="utf-8"))
    if (record.get("format_version") != 1 or record.get("model") != "dinov2_vits14"
            or record.get("weights_id") != "LVD142M" or record.get("url") != DINOV2_URL
            or record.get("license") != "Apache-2.0"
            or record.get("upstream_revision") != DINOV2_REVISION):
        raise ValueError("DINOv2 source manifest does not describe the allowed original weights")
    if record.get("sha256") != sha256_file(path) or record.get("bytes") != path.stat().st_size:
        raise ValueError("DINOv2 weight SHA256/size mismatch")
    documents = directory / "dinov2_vits14_documents"
    for name, expected in DINOV2_DOCUMENTS.items():
        if sha256_file(documents / name) != expected:
            raise ValueError(f"DINOv2 source document SHA256 mismatch: {name}")
    return path, manifest, documents
