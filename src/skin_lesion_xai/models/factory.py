from __future__ import annotations

from torch import nn

from skin_lesion_xai.config import ModelConfig
from skin_lesion_xai.models.backbones import FeatureBackbone, SingleBranchClassifier
from skin_lesion_xai.models.baselines import create_timm_classifier
from skin_lesion_xai.models.fusion import BidirectionalCrossAttentionFusion, ConcatFusion
from skin_lesion_xai.models.hybrid import HybridEffNetSwin


def build_model(cfg: ModelConfig, num_classes: int, pretrained: bool | None = None) -> nn.Module:
    """pretrained=None uses cfg.pretrained; pass False when loading a trained checkpoint."""
    pretrained = cfg.pretrained if pretrained is None else pretrained
    if cfg.arch == "timm":
        return create_timm_classifier(cfg.timm_name, num_classes, pretrained, cfg.drop_rate, cfg.drop_path_rate,
                                      cfg.match_pretrained_norm)
    if cfg.arch == "single_branch":
        backbone = FeatureBackbone(cfg.timm_name, pretrained, cfg.drop_path_rate, cfg.match_pretrained_norm)
        return SingleBranchClassifier(backbone, num_classes, cfg.drop_rate)
    if cfg.arch == "hybrid":
        local = FeatureBackbone(cfg.local_timm_name, pretrained, cfg.drop_path_rate, cfg.match_pretrained_norm)
        global_ = FeatureBackbone(cfg.global_timm_name, pretrained, cfg.drop_path_rate, cfg.match_pretrained_norm)
        if cfg.fusion == "cross_attention":
            fusion = BidirectionalCrossAttentionFusion(local.out_channels, global_.out_channels, cfg.fusion_dim,
                                                       cfg.fusion_layers, cfg.fusion_heads)
        elif cfg.fusion == "concat":
            fusion = ConcatFusion(local.out_channels, global_.out_channels)
        else:
            raise ValueError(f"unknown fusion {cfg.fusion!r} in model config {cfg.name!r}")
        return HybridEffNetSwin(local, global_, fusion, num_classes, cfg.drop_rate)
    raise ValueError(f"unknown arch {cfg.arch!r} in model config {cfg.name!r}")


def head_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Parameters of the freshly initialised classifier head (trained with the larger head_lr)."""
    if hasattr(model, "head_parameters"):
        return list(model.head_parameters())
    if hasattr(model, "get_classifier"):  # timm models
        return list(model.get_classifier().parameters())
    raise TypeError(f"cannot find the classifier head of {type(model).__name__}")
