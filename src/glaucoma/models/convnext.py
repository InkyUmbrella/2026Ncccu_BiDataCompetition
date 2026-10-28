import torch
from torch import nn
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

from .registry import register_model


@register_model("convnext_tiny")
class ConvNeXtClassifier(nn.Module):
    def __init__(self, use_age=False, hidden_dim=128, dropout=0.2, stochastic_depth_prob=0.1, weights=None):
        super().__init__()
        if type(use_age) is not bool or type(hidden_dim) is not int or hidden_dim < 1 or not 0 <= dropout < 1:
            raise ValueError("Invalid use_age, hidden_dim or dropout")
        if (type(stochastic_depth_prob) is bool or not isinstance(stochastic_depth_prob, (int, float))
                or not 0 <= stochastic_depth_prob <= 1):
            raise ValueError("stochastic_depth_prob must be in [0, 1]")
        # Construct the backbone first so B0/B1 start from identical image weights.
        # 0.1 is torchvision's historical default; keeping it here means configs and
        # resumed checkpoints written before this key existed keep their original
        # stochastic-depth behaviour. Pass an explicit 0.0 to disable it.
        if weights not in (None, "IMAGENET1K_V1"):
            raise ValueError("weights must be None or IMAGENET1K_V1")
        backbone = convnext_tiny(weights=ConvNeXt_Tiny_Weights[weights] if weights else None,
                                 stochastic_depth_prob=float(stochastic_depth_prob))
        self.encoder = nn.Sequential(backbone.features, backbone.avgpool,
                                     backbone.classifier[0], nn.Flatten(1))
        self.use_age = use_age
        self.stochastic_depth_prob = float(stochastic_depth_prob)
        self.age_encoder = nn.Sequential(nn.Linear(3, 16), nn.GELU(), nn.Linear(16, 8)) if use_age else None
        self.head = nn.Sequential(nn.Linear(768 + (8 if use_age else 0), hidden_dim),
                                  nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, image, age):
        features = self.encoder(image)
        if self.use_age:
            features = torch.cat((features, self.age_encoder(age)), dim=1)
        return self.head(features).squeeze(1)


@register_model("convnext_tiny_spatial_residual")
class ConvNeXtSpatialResidual(ConvNeXtClassifier):
    """Global classifier plus learned spatial pooling; starts at the base logits.

    Attention operates on the existing feature map: no second image encoder or
    annotations are needed. Zero output initialization preserves transferred
    predictions while allowing the local branch to learn during fine tuning.
    """

    def __init__(self, hidden_dim=128, dropout=0.2, stochastic_depth_prob=0.1,
                 weights=None, attention_heads=4, attention_dim=64):
        if type(attention_heads) is not int or not 1 <= attention_heads <= 16:
            raise ValueError("attention_heads must be an integer in [1, 16]")
        if type(attention_dim) is not int or not 1 <= attention_dim <= 256:
            raise ValueError("attention_dim must be an integer in [1, 256]")
        super().__init__(False, hidden_dim, dropout, stochastic_depth_prob, weights)
        self.spatial_residual = nn.ModuleDict({
            "attention": nn.Conv2d(768, attention_heads, 1),
            "projection": nn.Conv2d(768, attention_dim, 1),
            "head": nn.Sequential(nn.LayerNorm(attention_heads * attention_dim),
                                  nn.Linear(attention_heads * attention_dim, hidden_dim),
                                  nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1)),
        })
        nn.init.zeros_(self.spatial_residual["head"][-1].weight)
        nn.init.zeros_(self.spatial_residual["head"][-1].bias)

    def forward(self, image, age):
        spatial = self.encoder[0](image)
        global_features = self.encoder[3](self.encoder[2](self.encoder[1](spatial)))
        base = self.head(global_features).squeeze(1)
        attention = self.spatial_residual["attention"](spatial).flatten(2)
        # Compute the spatial softmax and weighted sum in float32 under AMP.
        attention = attention.float().softmax(dim=-1)
        values = self.spatial_residual["projection"](spatial).flatten(2).float()
        pooled = torch.bmm(attention, values.transpose(1, 2)).flatten(1)
        return base + self.spatial_residual["head"](pooled).squeeze(1)


@register_model("convnext_tiny_age_residual")
class ConvNeXtAgeResidual(ConvNeXtClassifier):
    """A B0 image classifier plus an initially zero age-only logit correction."""

    def __init__(self, hidden_dim=128, dropout=0.2, stochastic_depth_prob=0.1, weights=None):
        super().__init__(use_age=False, hidden_dim=hidden_dim, dropout=dropout,
                         stochastic_depth_prob=stochastic_depth_prob, weights=weights)
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
