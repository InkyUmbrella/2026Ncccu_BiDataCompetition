# Copyright (c) Meta Platforms, Inc. and affiliates.
# Adapted to a standalone, single-image ViT-S/14 and the project's classifier API.
# Apache-2.0; license and upstream attribution: licenses/dinov2/.
"""Original non-register DINOv2-S/14 state-dict layout, using native PyTorch SDPA.

Architecture/position interpolation follow facebookresearch/dinov2 revision
7764ea0f912e53c92e82eb78a2a1631e92725fc8. No Torch Hub code execution or downloads.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .pretrained import dinov2_source
from .registry import register_model


class _PatchEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3, 384, kernel_size=14, stride=14)

    def forward(self, image):
        return self.proj(image).flatten(2).transpose(1, 2)


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(384, 1152)
        self.proj = nn.Linear(384, 384)

    def forward(self, x):
        batch, tokens, channels = x.shape
        q, k, v = self.qkv(x).reshape(batch, tokens, 3, 6, 64).unbind(2)
        q, k, v = [item.transpose(1, 2) for item in (q, k, v)]
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        return self.proj(x.transpose(1, 2).contiguous().view(batch, tokens, channels))


class _Mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(384, 1536)
        self.fc2 = nn.Linear(1536, 384)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class _LayerScale(nn.Module):
    def __init__(self):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(384))

    def forward(self, x):
        return x * self.gamma


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(384, eps=1e-6)
        self.attn = _Attention()
        self.ls1 = _LayerScale()
        self.norm2 = nn.LayerNorm(384, eps=1e-6)
        self.mlp = _Mlp()
        self.ls2 = _LayerScale()

    def forward(self, x):
        x = x + self.ls1(self.attn(self.norm1(x)))
        return x + self.ls2(self.mlp(self.norm2(x)))


class DinoSmallBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = _PatchEmbed()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 384))
        # Official hub constructor uses img_size=518: 37x37 patches.
        self.pos_embed = nn.Parameter(torch.zeros(1, 1370, 384))
        self.mask_token = nn.Parameter(torch.zeros(1, 384))
        self.blocks = nn.ModuleList([_Block() for _ in range(12)])
        self.norm = nn.LayerNorm(384, eps=1e-6)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                nn.init.zeros_(module.bias)

    def position_encoding(self, tokens, height, width):
        if tokens.shape[1] == 1370 and height == width:
            return self.pos_embed
        patches = self.pos_embed[:, 1:].float().reshape(1, 37, 37, 384).permute(0, 3, 1, 2)
        # Preserve the original non-register model's offset and interpolation.
        patches = F.interpolate(patches, scale_factor=((height // 14 + 0.1) / 37,
                                                       (width // 14 + 0.1) / 37),
                                mode="bicubic", antialias=False)
        if patches.shape[-2:] != (height // 14, width // 14):
            raise ValueError("Unexpected DINOv2 position embedding shape")
        patches = patches.permute(0, 2, 3, 1).reshape(1, -1, 384)
        return torch.cat((self.pos_embed[:, :1].float(), patches), dim=1).to(tokens.dtype)

    def forward(self, image):
        height, width = image.shape[-2:]
        if height % 14 or width % 14:
            raise ValueError("DINOv2 inputs must be multiples of patch size 14")
        tokens = self.patch_embed(image)
        tokens = torch.cat((self.cls_token.expand(image.shape[0], -1, -1), tokens), dim=1)
        tokens = tokens + self.position_encoding(tokens, height, width)
        for block in self.blocks:
            tokens = block(tokens)
        return self.norm(tokens)[:, 0]  # x_norm_clstoken; no additional L2 normalization.


@register_model("dinov2_vits14")
class DinoSmallClassifier(nn.Module):
    def __init__(self, hidden_dim=128, dropout=0.2, use_age=False, weights=None):
        super().__init__()
        if use_age is not False or type(hidden_dim) is not int or hidden_dim < 1 or not 0 <= dropout < 1:
            raise ValueError("DINOv2 requires use_age=False, positive hidden_dim and dropout in [0, 1)")
        if weights not in (None, "LVD142M"):
            raise ValueError("DINOv2 weights must be None or LVD142M")
        self.encoder = DinoSmallBackbone()
        if weights:
            path, _, _ = dinov2_source()
            state = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(state, dict) or any(not isinstance(t, torch.Tensor)
                                                  or not torch.isfinite(t).all() for t in state.values()):
                raise ValueError("DINOv2 backbone must contain finite tensors only")
            self.encoder.load_state_dict(state, strict=True)
        self.head = nn.Sequential(nn.Linear(384, hidden_dim), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(hidden_dim, 1))

    def forward(self, image, age):
        return self.head(self.encoder(image)).squeeze(1)
