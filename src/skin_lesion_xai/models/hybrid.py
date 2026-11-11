"""HybridEffNetSwin: EfficientNetV2-S (local branch) + Swin-T (global branch) + fusion + linear head.

Both branches see the same 224x224 image and return 7x7 grids (backbones.py); the fusion module
(fusion.py) turns them into one vector; the head is zero-initialised (ViT App. B.1.1). Grad-CAM can
hook local_backbone / global_backbone outputs; cross-attention maps come from fusion.attention_maps().
"""
from __future__ import annotations

import itertools

import torch
from torch import nn

from skin_lesion_xai.models.backbones import FeatureBackbone


class HybridEffNetSwin(nn.Module):
    def __init__(self, local_backbone: FeatureBackbone, global_backbone: FeatureBackbone, fusion: nn.Module,
                 num_classes: int, drop_rate: float = 0.0):
        super().__init__()
        self.local_backbone = local_backbone
        self.global_backbone = global_backbone
        self.fusion = fusion
        self.dropout = nn.Dropout(drop_rate)
        self.head = nn.Linear(fusion.out_dim, num_classes)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def set_grad_checkpointing(self, enable: bool = True) -> None:
        self.local_backbone.set_grad_checkpointing(enable)
        self.global_backbone.set_grad_checkpointing(enable)

    def head_parameters(self):
        """Newly initialised parameters (fusion + head), trained with head_lr."""
        return itertools.chain(self.fusion.parameters(), self.head.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fused = self.fusion(self.local_backbone(x), self.global_backbone(x))
        return self.head(self.dropout(fused))
