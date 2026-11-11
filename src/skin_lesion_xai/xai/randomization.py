"""Cascading model-parameter randomisation (Adebayo et al. 2018, Related_papers/Sanity Checks for Saliency Maps).

The weights are re-initialised block by block from the top (head) to the bottom (stem), cumulatively, and
the explanation after each step is compared with the trained one (similarity.py). A map that stays similar
does not depend on what the model learned. As in the paper's code (code/src/utils.py, reinitlayerlist), each
trainable parameter gets its initialiser again and buffers (BatchNorm running statistics) are kept.

One rule for both architectures: PyTorch's default initialiser (reset_parameters), affine norms back to
1 / 0, ConvNeXt layer scale back to timm's 1e-6, the fusion's position / type embeddings back to
trunc_normal(0.02). The zero-initialised heads get PyTorch's default instead (all-zero weights would give
all-zero gradients). Any parameter not covered by these rules raises an error.

Steps are grouped by output resolution so the two models are comparable (224 px input):
  ConvNeXt-T: head, stage 4 (7x7), stage 3 (14x14), stage 2 (28x28), stage 1 (56x56), stem.
  Hybrid: head, fusion, then both branches together at 7, 14, 28, 56 px grids and their stems
  (EfficientNetV2-S blocks 5 / 4-3 / 2 / 1 / 0, Swin-T layers 3 / 2 / 1 / 0).
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm

from skin_lesion_xai.models.hybrid import HybridEffNetSwin

_LOCAL, _GLOBAL = "local_backbone.model", "global_backbone.model"
HYBRID_CASCADE = [
    ("head", ["head"]),
    ("fusion", ["fusion"]),
    ("grid7", [f"{_LOCAL}.blocks.5", f"{_LOCAL}.conv_head", f"{_LOCAL}.bn2", f"{_GLOBAL}.layers.3", f"{_GLOBAL}.norm"]),
    ("grid14", [f"{_LOCAL}.blocks.4", f"{_LOCAL}.blocks.3", f"{_GLOBAL}.layers.2"]),
    ("grid28", [f"{_LOCAL}.blocks.2", f"{_GLOBAL}.layers.1"]),
    ("grid56", [f"{_LOCAL}.blocks.1", f"{_GLOBAL}.layers.0"]),
    ("stem", [f"{_LOCAL}.blocks.0", f"{_LOCAL}.conv_stem", f"{_LOCAL}.bn1", f"{_GLOBAL}.patch_embed"]),
]
CONVNEXT_CASCADE = [
    ("head", ["head"]),
    ("grid7", ["stages.3"]),
    ("grid14", ["stages.2"]),
    ("grid28", ["stages.1"]),
    ("grid56", ["stages.0"]),
    ("stem", ["stem"]),
]


def cascade(model: nn.Module) -> list[tuple[str, list[str]]]:
    from timm.models.convnext import ConvNeXt

    if isinstance(model, HybridEffNetSwin):
        return HYBRID_CASCADE
    if isinstance(model, ConvNeXt):
        return CONVNEXT_CASCADE
    raise TypeError(f"no randomisation cascade for {type(model).__name__}")


@torch.no_grad()
def reinitialize(block: nn.Module) -> None:
    from timm.models.convnext import ConvNeXtBlock

    from skin_lesion_xai.models.fusion import BidirectionalCrossAttentionFusion

    for module in block.modules():
        own = dict(module.named_parameters(recurse=False))
        if not own:
            continue
        if isinstance(module, _BatchNorm):  # affine only; running statistics are buffers and stay
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif hasattr(module, "reset_parameters"):
            module.reset_parameters()
        elif isinstance(module, ConvNeXtBlock) and set(own) == {"gamma"}:
            module.gamma.fill_(1e-6)
        elif isinstance(module, BidirectionalCrossAttentionFusion) and set(own) == {"pos_embed", "type_embed"}:
            nn.init.trunc_normal_(module.pos_embed, std=0.02)
            nn.init.trunc_normal_(module.type_embed, std=0.02)
        else:
            raise TypeError(f"no re-initialisation rule for {type(module).__name__} parameters {sorted(own)}")


def randomize_step(model: nn.Module, paths: list[str]) -> None:
    for path in paths:
        reinitialize(model.get_submodule(path))
