"""Strict JSON configuration, shared by training and checkpoint inference."""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path


@dataclass
class DataConfig:
    root: str = "dataset/glaucoma_first"
    image_size: int = 384
    num_workers: int = 4
    verify_images: bool = True
    crop_black_border: bool = False
    crop_threshold: int = 10
    crop_margin: float = 0.03
    # "full" is the historical, exact decode path. "draft" lets Pillow's JPEG
    # DCT scaling decode to just above the model input (training/inference parity
    # is carried by the checkpoint config).
    decode: str = "full"


@dataclass
class ModelConfig:
    name: str = "convnext_tiny"
    params: dict = field(default_factory=lambda: {"use_age": False, "hidden_dim": 128, "dropout": 0.2})


@dataclass
class TrainConfig:
    epochs: int = 60
    batch_size: int = 16
    accumulation_steps: int = 4
    learning_rate: float = 3e-4
    min_learning_rate: float = 1e-6
    warmup_epochs: int = 5
    weight_decay: float = 0.05
    pos_weight: float = 5.0
    grad_clip: float = 1.0
    amp: bool = True
    positive_fraction: float | None = None
    freeze_image_epochs: int = 0
    image_lr_scale: float = 1.0
    # Default-off training upgrades; null/0 keeps the original numerics.
    ema_decay: float | None = None
    ema_warmup_steps: int = 0
    mixup_alpha: float | None = None
    cutmix_alpha: float | None = None
    label_smoothing: float = 0.0


@dataclass
class Config:
    experiment: str = "b0_image"
    seed: int = 42
    device: str = "cuda"
    output_root: str = "outputs/baseline"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def validate(self):
        import math

        if not self.experiment or any(c in self.experiment for c in "/\\") or self.experiment in (".", ".."):
            raise ValueError("experiment must be a single directory name")
        for name, value in {"image_size": self.data.image_size, "epochs": self.train.epochs,
                            "batch_size": self.train.batch_size,
                            "accumulation_steps": self.train.accumulation_steps}.items():
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.data.image_size < 32:
            raise ValueError("image_size must be at least 32")
        if type(self.data.crop_black_border) is not bool:
            raise ValueError("crop_black_border must be a boolean")
        if type(self.data.crop_threshold) is not int or not 0 <= self.data.crop_threshold <= 255:
            raise ValueError("crop_threshold must be an integer in [0, 255]")
        if not math.isfinite(self.data.crop_margin) or not 0 <= self.data.crop_margin <= 0.5:
            raise ValueError("crop_margin must be in [0, 0.5]")
        if self.data.decode not in ("full", "draft"):
            raise ValueError('decode must be "full" or "draft"')
        fraction = self.train.positive_fraction
        if fraction is not None and (not math.isfinite(fraction) or not 0 < fraction < 1):
            raise ValueError("positive_fraction must be in (0, 1)")
        if type(self.train.freeze_image_epochs) is not int or not 0 <= self.train.freeze_image_epochs < self.train.epochs:
            raise ValueError("freeze_image_epochs must be in [0, epochs)")
        if not math.isfinite(self.train.image_lr_scale) or not 0 < self.train.image_lr_scale <= 1:
            raise ValueError("image_lr_scale must be in (0, 1]")
        if self.train.ema_decay is not None and (
                not math.isfinite(self.train.ema_decay) or not 0 < self.train.ema_decay < 1):
            raise ValueError("ema_decay must be in (0, 1) or null")
        if type(self.train.ema_warmup_steps) is not int or self.train.ema_warmup_steps < 0:
            raise ValueError("ema_warmup_steps must be a nonnegative integer")
        for name in ("mixup_alpha", "cutmix_alpha"):
            value = getattr(self.train, name)
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive or null")
        if not math.isfinite(self.train.label_smoothing) or not 0 <= self.train.label_smoothing <= 0.5:
            raise ValueError("label_smoothing must be in [0, 0.5]")
        if self.train.freeze_image_epochs and self.model.name != "convnext_tiny_age_residual":
            raise ValueError("freeze_image_epochs requires convnext_tiny_age_residual")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        if type(self.data.num_workers) is not int or self.data.num_workers < 0:
            raise ValueError("num_workers must be a nonnegative integer")
        if type(self.train.warmup_epochs) is not int or not 0 <= self.train.warmup_epochs < self.train.epochs:
            raise ValueError("warmup_epochs must be in [0, epochs)")
        for name in ("learning_rate", "min_learning_rate", "pos_weight", "grad_clip"):
            value = getattr(self.train, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.train.weight_decay) or self.train.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if self.train.min_learning_rate > self.train.learning_rate:
            raise ValueError("min_learning_rate exceeds learning_rate")
        if not isinstance(self.model.params, dict):
            raise ValueError("model.params must be an object")
        for value in (self.train.amp, self.data.verify_images):
            if type(value) is not bool:
                raise ValueError("amp and verify_images must be booleans")
        return self

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        for key, schema in (("data", DataConfig), ("model", ModelConfig), ("train", TrainConfig)):
            if key in values:
                values[key] = schema(**values[key])
        return cls(**values).validate()

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
