import torch
from torch import nn
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

from .registry import register_model


@register_model("convnext_tiny")
class ConvNeXtClassifier(nn.Module):
    def __init__(self, use_age=False, hidden_dim=128, dropout=0.2, weights=None):
        super().__init__()
        if type(use_age) is not bool or type(hidden_dim) is not int or hidden_dim < 1 or not 0 <= dropout < 1:
            raise ValueError("Invalid use_age, hidden_dim or dropout")
        # Construct the backbone first so B0/B1 start from identical image weights.
        if weights not in (None, "IMAGENET1K_V1"):
            raise ValueError("weights must be None or IMAGENET1K_V1")
        backbone = convnext_tiny(weights=ConvNeXt_Tiny_Weights[weights] if weights else None)
        self.encoder = nn.Sequential(backbone.features, backbone.avgpool,
                                     backbone.classifier[0], nn.Flatten(1))
        self.use_age = use_age
        self.age_encoder = nn.Sequential(nn.Linear(3, 16), nn.GELU(), nn.Linear(16, 8)) if use_age else None
        self.head = nn.Sequential(nn.Linear(768 + (8 if use_age else 0), hidden_dim),
                                  nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, image, age):
        features = self.encoder(image)
        if self.use_age:
            features = torch.cat((features, self.age_encoder(age)), dim=1)
        return self.head(features).squeeze(1)


@register_model("convnext_tiny_age_residual")
class ConvNeXtAgeResidual(ConvNeXtClassifier):
    """A B0 image classifier plus an initially zero age-only logit correction."""

    def __init__(self, hidden_dim=128, dropout=0.2, weights=None):
        super().__init__(use_age=False, hidden_dim=hidden_dim, dropout=dropout, weights=weights)
        self.age_residual = nn.Sequential(nn.Linear(3, 16), nn.GELU(), nn.Linear(16, 1))
        nn.init.zeros_(self.age_residual[-1].weight)
        nn.init.zeros_(self.age_residual[-1].bias)
        self.image_frozen = False

    def set_image_frozen(self, frozen):
        self.image_frozen = frozen
        for module in (self.encoder, self.head):
            module.requires_grad_(not frozen)

    def train(self, mode=True):
        super().train(mode)
        if self.image_frozen:
            self.encoder.eval()
            self.head.eval()
        return self

    def forward(self, image, age):
        return super().forward(image, age) + self.age_residual(age).squeeze(1)
