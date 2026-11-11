"""Feature backbones returning spatial feature grids (B, C, H, W).

At 224x224 input both branches of the hybrid give a 7x7 grid (stride 32):
  - EfficientNetV2-S: forward_features -> (B, 1280, 7, 7), after conv_head + BN + activation
  - Swin-T:           forward_features -> (B, 7, 7, 768) channels-last, permuted here to (B, 768, 7, 7)
Milestone 3 builds the cross-attention fusion on these grids; Grad-CAM can hook the backbone output.

The data pipeline normalises every image with ImageNet mean/std, but some checkpoints were pretrained
with mean = std = 0.5 (tf_efficientnetv2_s.in21k_ft_in1k, vit_base_patch16_224.augreg2_in21k_ft_in1k).
match_pretrained_norm=True re-normalises the input to the checkpoint's own statistics (InputRenorm).
It is opt-in so runs trained without it (all Milestone 2 baselines) reload and behave unchanged.
"""
from __future__ import annotations

import timm
import torch
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch import nn


class InputRenorm(nn.Module):
    """Maps an ImageNet-normalised image to a model's pretrained normalisation (identity if they match).
    The buffers are not saved, so checkpoints are identical with or without it."""

    def __init__(self, target_mean, target_std, source_mean=IMAGENET_DEFAULT_MEAN, source_std=IMAGENET_DEFAULT_STD):
        super().__init__()
        src_m, src_s = torch.tensor(source_mean), torch.tensor(source_std)
        tgt_m, tgt_s = torch.tensor(target_mean), torch.tensor(target_std)
        # (x * src_s + src_m - tgt_m) / tgt_s
        self.register_buffer("scale", (src_s / tgt_s).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("shift", ((src_m - tgt_m) / tgt_s).view(1, 3, 1, 1), persistent=False)

    @classmethod
    def for_model(cls, model: nn.Module) -> "InputRenorm":
        cfg = model.pretrained_cfg
        return cls(cfg["mean"], cfg["std"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale + self.shift


class FeatureBackbone(nn.Module):
    def __init__(self, timm_name: str, pretrained: bool = True, drop_path_rate: float = 0.0,
                 match_pretrained_norm: bool = False):
        super().__init__()
        self.model = timm.create_model(
            timm_name, pretrained=pretrained, num_classes=0, global_pool="", drop_path_rate=drop_path_rate
        )
        self.input_renorm = InputRenorm.for_model(self.model) if match_pretrained_norm else nn.Identity()
        self.out_channels: int = self.model.num_features
        # timm Swin returns NHWC features; CNNs return NCHW
        self.channels_last_output = timm_name.startswith("swin")

    def set_grad_checkpointing(self, enable: bool = True) -> None:
        self.model.set_grad_checkpointing(enable)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.model.forward_features(self.input_renorm(x))
        if self.channels_last_output:
            features = features.permute(0, 3, 1, 2).contiguous()
        return features


class SingleBranchClassifier(nn.Module):
    """Backbone + global average pooling + dropout + linear (zero-init, ViT App. B.1.1)."""

    def __init__(self, backbone: FeatureBackbone, num_classes: int, drop_rate: float = 0.0):
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(drop_rate)
        self.head = nn.Linear(backbone.out_channels, num_classes)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def set_grad_checkpointing(self, enable: bool = True) -> None:
        self.backbone.set_grad_checkpointing(enable)

    def head_parameters(self):
        return self.head.parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.backbone(x).mean(dim=(2, 3))
        return self.head(self.dropout(pooled))
