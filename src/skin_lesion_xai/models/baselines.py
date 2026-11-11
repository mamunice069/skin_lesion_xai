"""Baselines from the proposal (slide 9): ResNet-50, ConvNeXt-T, ViT-B/16 (timm classifiers),
and the single-branch Swin-T / EfficientNetV2-S models (backbones.py)."""
from __future__ import annotations

import timm
from torch import nn

from skin_lesion_xai.models.backbones import InputRenorm


def create_timm_classifier(timm_name: str, num_classes: int, pretrained: bool = True,
                           drop_rate: float = 0.0, drop_path_rate: float = 0.0,
                           match_pretrained_norm: bool = False) -> nn.Module:
    """timm model with a freshly initialised num_classes head (pretrained head is dropped)."""
    model = timm.create_model(
        timm_name, pretrained=pretrained, num_classes=num_classes,
        drop_rate=drop_rate, drop_path_rate=drop_path_rate,
    )
    if match_pretrained_norm:
        # a pre-hook keeps the timm module (and its state_dict keys / get_classifier) unchanged
        model.input_renorm = InputRenorm.for_model(model)
        model.register_forward_pre_hook(lambda module, args: (module.input_renorm(args[0]), *args[1:]))
    return model
