"""Official torchvision EfficientNetV2-S with the competition binary head."""

import math

import torch
from torch import nn
from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s

from .registry import register_model


@register_model("efficientnet_v2_s")
class EfficientNetV2Classifier(nn.Module):
    def __init__(self, use_age=False, hidden_dim=128, dropout=0.2,
                 stochastic_depth_prob=0.2, weights=None):
        super().__init__()
        if type(use_age) is not bool or type(hidden_dim) is not int or hidden_dim < 1:
            raise ValueError("Invalid use_age or hidden_dim")
        for name, value, upper in (("dropout", dropout, 1),
                                   ("stochastic_depth_prob", stochastic_depth_prob, 1)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Invalid {name}")
            if not 0 <= value < upper:
                raise ValueError(f"{name} must be in [0, 1)")
        if weights not in (None, "IMAGENET1K_V1"):
            raise ValueError("weights must be None or IMAGENET1K_V1")
        backbone = efficientnet_v2_s(
            weights=EfficientNet_V2_S_Weights[weights] if weights else None,
            stochastic_depth_prob=stochastic_depth_prob)
        width = backbone.classifier[-1].in_features
        self.encoder = nn.Sequential(backbone.features, backbone.avgpool, nn.Flatten(1))
        self.use_age = use_age
        self.age_encoder = nn.Sequential(nn.Linear(3, 16), nn.GELU(), nn.Linear(16, 8)) if use_age else None
        self.head = nn.Sequential(nn.Linear(width + (8 if use_age else 0), hidden_dim),
                                  nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, image, age):
        features = self.encoder(image)
        if self.use_age:
            features = torch.cat((features, self.age_encoder(age)), dim=1)
        return self.head(features).squeeze(1)
