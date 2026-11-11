"""Attention-based maps (class-agnostic, unlike Grad-CAM).

ViT-B/16 attention rollout (Dosovitskiy et al., App. D.8, after Abnar & Zuidema 2020): the attention of each
layer is averaged over heads, the residual connection is accounted for as 0.5 * A + 0.5 * I (rows
renormalised), and the layers are multiplied from first to last. The class token's row over the patch
tokens is the 14x14 map.

Hybrid cross-attention maps: fusion.attention_maps() holds, per layer, the local-query -> global-key and
global-query -> local-key attention (B, heads, 49, 49). Averaged over layers, heads and queries this gives,
for each direction, how much attention each of the 49 key locations receives (each map sums to 1):
  - "xattn_local_to_global": where on the Swin grid the EfficientNetV2 tokens look;
  - "xattn_global_to_local": where on the EfficientNetV2 grid the Swin tokens look.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from skin_lesion_xai.models.fusion import BidirectionalCrossAttentionFusion
from skin_lesion_xai.models.hybrid import HybridEffNetSwin


def rollout(attentions: list[torch.Tensor], prefix_tokens: int = 1) -> torch.Tensor:
    """attentions: per layer (B, heads, N, N), first layer first. Returns (B, side, side)."""
    joint = None
    for attn in attentions:
        a = attn.float().mean(dim=1)
        a = 0.5 * a + 0.5 * torch.eye(a.shape[-1], device=a.device)
        a = a / a.sum(dim=-1, keepdim=True)
        joint = a if joint is None else a @ joint
    cls_row = joint[:, 0, prefix_tokens:]
    side = int(round(cls_row.shape[1] ** 0.5))
    return cls_row.reshape(-1, side, side)


@torch.no_grad()
def vit_attention_rollout(model: nn.Module, x: torch.Tensor) -> np.ndarray:
    """(B, 14, 14) rollout maps of a timm VisionTransformer."""
    attns: list[torch.Tensor] = []
    blocks = list(model.blocks)
    fused = [b.attn.fused_attn for b in blocks]
    handles = [b.attn.attn_drop.register_forward_hook(lambda m, args, out: attns.append(out)) for b in blocks]
    try:
        for b in blocks:  # the fused kernel never forms the attention matrix
            b.attn.fused_attn = False
        model.eval()
        with torch.autocast(device_type=x.device.type, enabled=False):
            model(x.float())
    finally:
        for b, f in zip(blocks, fused):
            b.attn.fused_attn = f
        for h in handles:
            h.remove()
    if len(attns) != len(blocks):
        raise RuntimeError(f"captured {len(attns)} attention maps for {len(blocks)} blocks")
    return rollout(attns, model.num_prefix_tokens).cpu().numpy()


def key_attention(attn: torch.Tensor) -> torch.Tensor:
    """(B, heads, Nq, Nk) -> (B, Nk): mean over heads and queries."""
    return attn.float().mean(dim=(1, 2))


@torch.no_grad()
def cross_attention_maps(model: HybridEffNetSwin, x: torch.Tensor) -> dict[str, np.ndarray]:
    """{"xattn_local_to_global", "xattn_global_to_local"}: (B, 7, 7) each."""
    fusion = model.fusion
    if not isinstance(fusion, BidirectionalCrossAttentionFusion):
        raise TypeError("cross-attention maps need the cross-attention fusion")
    fusion.set_store_attention(True)
    try:
        model.eval()
        with torch.autocast(device_type=x.device.type, enabled=False):
            model(x.float())
        layers = fusion.attention_maps()
    finally:
        fusion.set_store_attention(False)
    side = fusion.grid_size
    out = {}
    for direction in ("local_to_global", "global_to_local"):
        mean = torch.stack([key_attention(layer[direction]) for layer in layers]).mean(dim=0)
        out[f"xattn_{direction}"] = mean.reshape(-1, side, side).cpu().numpy()
    return out


def attention_methods(model: nn.Module) -> list[str]:
    from timm.models.vision_transformer import VisionTransformer

    if isinstance(model, VisionTransformer):
        return ["rollout"]
    if isinstance(model, HybridEffNetSwin) and isinstance(model.fusion, BidirectionalCrossAttentionFusion):
        return ["xattn_local_to_global", "xattn_global_to_local"]
    return []


def attention_maps(model: nn.Module, x: torch.Tensor) -> dict[str, np.ndarray]:
    """All attention maps defined for this model ({} for CNNs, Swin-T and the concat hybrid)."""
    methods = attention_methods(model)
    if methods == ["rollout"]:
        return {"rollout": vit_attention_rollout(model, x)}
    if methods:
        return cross_attention_maps(model, x)
    return {}
