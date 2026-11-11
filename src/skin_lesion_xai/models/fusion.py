"""Fusion of the two 7x7 feature grids of the hybrid (local = EfficientNetV2, global = Swin).

BidirectionalCrossAttentionFusion (the proposed model):
  - both grids are projected to `dim` channels (1x1 conv) and flattened to 49 tokens each;
  - both branches have stride 32 at 224 px, so token i is the same image location in both. They share one
    learnable position embedding (location) and get a branch-type embedding (local / global);
  - each layer updates both streams from the layer's input: local tokens query the global tokens and
    global tokens query the local tokens. Pre-norm attention + MLP with residuals, as the ViT encoder block
    (Dosovitskiy et al. §3.1, Eq. 2-3), applied in token space after the backbones as in GC-ViT-E §3.3.2;
  - output: mean of each stream, concatenated (2 * dim).
  Attention weights are kept when store_attention is on (for the XAI attention maps), otherwise the fused
  scaled_dot_product_attention kernel is used.

ConcatFusion (ablation): global average pool of each grid, concatenated, with no attention between the branches.
(Mahesh et al. §3.5 concatenate image features with patient-metadata features, not two image branches; their
Conclusions name attention-based interactions across modalities as future work.)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from timm.layers import DropPath, Mlp, trunc_normal_
from torch import nn


class CrossAttention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        if dim % heads:
            raise ValueError(f"dim {dim} is not divisible by heads {heads}")
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.proj = nn.Linear(dim, dim)
        self.store_attention = False
        self.attention: torch.Tensor | None = None  # (B, heads, N_query, N_context), when stored

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        q = self.q(x).reshape(b, n, self.heads, c // self.heads).transpose(1, 2)
        k, v = self.kv(context).reshape(b, context.shape[1], 2, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        if self.store_attention:
            attn = ((q * self.scale) @ k.transpose(-2, -1)).softmax(dim=-1)
            self.attention = attn
            out = attn @ v
        else:
            out = F.scaled_dot_product_attention(q, k, v)
        return self.proj(out.transpose(1, 2).reshape(b, n, c))


class CrossAttentionBlock(nn.Module):
    """x <- x + Attn(LN(x), LN(context));  x <- x + MLP(LN(x))."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0, drop_path: float = 0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = CrossAttention(dim, heads)
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm_q(x), self.norm_kv(context)))
        return x + self.drop_path(self.mlp(self.norm_mlp(x)))


class BidirectionalCrossAttentionFusion(nn.Module):
    def __init__(self, local_channels: int, global_channels: int, dim: int = 384, layers: int = 2,
                 heads: int = 8, grid_size: int = 7, drop_path: float = 0.0):
        super().__init__()
        self.grid_size = grid_size
        self.local_proj = nn.Conv2d(local_channels, dim, kernel_size=1)
        self.global_proj = nn.Conv2d(global_channels, dim, kernel_size=1)
        self.pos_embed = nn.Parameter(torch.zeros(1, grid_size * grid_size, dim))
        self.type_embed = nn.Parameter(torch.zeros(2, dim))  # 0 = local, 1 = global
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.type_embed, std=0.02)
        self.local_blocks = nn.ModuleList(CrossAttentionBlock(dim, heads, drop_path=drop_path) for _ in range(layers))
        self.global_blocks = nn.ModuleList(CrossAttentionBlock(dim, heads, drop_path=drop_path) for _ in range(layers))
        self.local_norm = nn.LayerNorm(dim)
        self.global_norm = nn.LayerNorm(dim)
        self.out_dim = 2 * dim

    def set_store_attention(self, enable: bool = True) -> None:
        for block in [*self.local_blocks, *self.global_blocks]:
            block.attn.store_attention = enable
            block.attn.attention = None

    def attention_maps(self) -> list[dict[str, torch.Tensor]]:
        """Per layer: 'local_to_global' (local queries over global tokens) and 'global_to_local',
        each (B, heads, 49, 49), from the last forward pass with store_attention on."""
        return [{"local_to_global": lb.attn.attention, "global_to_local": gb.attn.attention}
                for lb, gb in zip(self.local_blocks, self.global_blocks)]

    def _tokens(self, grid: torch.Tensor, proj: nn.Conv2d, branch: int) -> torch.Tensor:
        if grid.shape[-2:] != (self.grid_size, self.grid_size):
            raise ValueError(f"expected a {self.grid_size}x{self.grid_size} grid, got {tuple(grid.shape[-2:])}")
        return proj(grid).flatten(2).transpose(1, 2) + self.pos_embed + self.type_embed[branch]

    def forward(self, local_grid: torch.Tensor, global_grid: torch.Tensor) -> torch.Tensor:
        local_tokens = self._tokens(local_grid, self.local_proj, 0)
        global_tokens = self._tokens(global_grid, self.global_proj, 1)
        for local_block, global_block in zip(self.local_blocks, self.global_blocks):
            local_tokens, global_tokens = local_block(local_tokens, global_tokens), global_block(global_tokens, local_tokens)
        return torch.cat([self.local_norm(local_tokens).mean(dim=1), self.global_norm(global_tokens).mean(dim=1)], dim=1)


class ConcatFusion(nn.Module):
    def __init__(self, local_channels: int, global_channels: int):
        super().__init__()
        self.out_dim = local_channels + global_channels

    def forward(self, local_grid: torch.Tensor, global_grid: torch.Tensor) -> torch.Tensor:
        return torch.cat([local_grid.mean(dim=(2, 3)), global_grid.mean(dim=(2, 3))], dim=1)
