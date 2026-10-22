"""Strict JSON configuration for self-supervised pretraining.

Kept separate from the supervised :class:`glaucoma.config.Config` so that every
existing experiment/checkpoint stays byte-for-byte compatible.
"""

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path


@dataclass
class SSLDataConfig:
    root: str = "dataset/glaucoma_first"
    cache: str | None = None
    image_size: int = 384
    ssl_size: int = 224
    num_workers: int = 8
    crop_black_border: bool = True
    crop_threshold: int = 10
    crop_margin: float = 0.03
    decode: str = "full"
    # Random-resized-crop floor for the two views. The plan suggested 0.55 to
    # protect the optic disc; the task brief fixes 0.2 as the MoCo-v2 default.
    min_crop_scale: float = 0.2


@dataclass
class SSLTrainConfig:
    epochs: int = 200
    batch_size: int = 256
    learning_rate: float = 0.03
    momentum: float = 0.9
    weight_decay: float = 0.0001
    warmup_epochs: int = 10
    temperature: float = 0.2
    queue_size: int = 65536
    moco_momentum: float = 0.999
    hidden_dim: int = 2048
    projection_dim: int = 128
    amp: bool = True
    save_every: int = 10
    probe_every: int = 50
    grad_clip: float = 0.0
    # Frozen linear probe used for the collapse/go-no-go signal. It is fitted
    # and evaluated on disjoint subsets of the *train* split only.
    probe_fit_size: int = 8000
    probe_eval_size: int = 4000


@dataclass
class SSLConfig:
    experiment: str = "ssl_moco_v2"
    seed: int = 42
    device: str = "cuda"
    output_root: str = "outputs/ssl"
    data: SSLDataConfig = field(default_factory=SSLDataConfig)
    train: SSLTrainConfig = field(default_factory=SSLTrainConfig)

    def validate(self):
        if not self.experiment or any(c in self.experiment for c in "/\\") or self.experiment in (".", ".."):
            raise ValueError("experiment must be a single directory name")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        if type(self.data.image_size) is not int or self.data.image_size < 32:
            raise ValueError("image_size must be an integer >= 32")
        if type(self.data.ssl_size) is not int or not 96 <= self.data.ssl_size <= 384:
            raise ValueError("ssl_size must be an integer in [96, 384]")
        if self.data.ssl_size > self.data.image_size:
            raise ValueError("ssl_size must not exceed the cache image_size")
        if type(self.data.num_workers) is not int or self.data.num_workers < 0:
            raise ValueError("num_workers must be a nonnegative integer")
        if type(self.data.crop_black_border) is not bool:
            raise ValueError("crop_black_border must be a boolean")
        if type(self.data.crop_threshold) is not int or not 0 <= self.data.crop_threshold <= 255:
            raise ValueError("crop_threshold must be an integer in [0, 255]")
        if not math.isfinite(self.data.crop_margin) or not 0 <= self.data.crop_margin <= 0.5:
            raise ValueError("crop_margin must be in [0, 0.5]")
        if not math.isfinite(self.data.min_crop_scale) or not 0 < self.data.min_crop_scale <= 1:
            raise ValueError("min_crop_scale must be in (0, 1]")
        if self.data.decode not in ("full", "draft"):
            raise ValueError('decode must be "full" or "draft"')
        train = self.train
        for name in ("epochs", "batch_size", "queue_size", "hidden_dim", "projection_dim",
                     "save_every", "probe_every"):
            value = getattr(train, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(train.warmup_epochs) is not int or not 0 <= train.warmup_epochs < train.epochs:
            raise ValueError("warmup_epochs must be in [0, epochs)")
        if not math.isfinite(train.learning_rate) or train.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(train.momentum) or not 0 <= train.momentum < 1:
            raise ValueError("momentum must be in [0, 1)")
        if not math.isfinite(train.weight_decay) or train.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if not math.isfinite(train.temperature) or not 0 < train.temperature < 1:
            raise ValueError("temperature must be in (0, 1)")
        if not math.isfinite(train.moco_momentum) or not 0 < train.moco_momentum < 1:
            raise ValueError("moco_momentum must be in (0, 1)")
        if not math.isfinite(train.grad_clip) or train.grad_clip < 0:
            raise ValueError("grad_clip must be finite and nonnegative")
        for name in ("amp",):
            if type(getattr(train, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        for name in ("probe_fit_size", "probe_eval_size"):
            if type(getattr(train, name)) is not int or getattr(train, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        return self

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        for key, schema in (("data", SSLDataConfig), ("train", SSLTrainConfig)):
            if key in values:
                values[key] = schema(**values[key])
        return cls(**values).validate()

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
