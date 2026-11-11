"""Mask-guided fine-tuning losses for the hybrid (Part B, EXPLORATORY): "right for the right reasons".

The hybrid's head is linear on the token means of the two fusion streams (fusion.py: cat of
local_norm(tokens).mean, global_norm(tokens).mean), so d logit_c / d token_i = W_c / 49 for every token and
its fused Grad-CAM (gradcam.py, "gradcam_fused") is exactly CAM / 49 with
    CAM_c(i) = W_c[:d] . local_norm(tokens)_i + W_c[d:] . global_norm(tokens)_i
(class activation mapping, Zhou et al. 2016; external). This map is differentiable at the cost of a forward
pass, with no second-order gradients, and it is what the losses below act on:
  outside_share   share of the positive CAM of the true class lying outside the lesion mask (mask area-pooled
                  to the 7x7 token grid); 0 = all of the map inside the lesion
  consistency     1 - cosine similarity of the positive CAMs of an image and of the same image with Gaussian
                  noise (std on the 0-1 pixel scale, as in the stability check)
Masks come from the train split only (scripts/finetune_mask.py).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from skin_lesion_xai.data.transforms import IMAGENET_MEAN, IMAGENET_STD
from skin_lesion_xai.models.fusion import BidirectionalCrossAttentionFusion
from skin_lesion_xai.models.hybrid import HybridEffNetSwin


class FusedCAM:
    """Forward pass that also returns the differentiable fused CAM (B, side, side) of the given classes."""

    def __init__(self, model: HybridEffNetSwin):
        if not (isinstance(model, HybridEffNetSwin) and isinstance(model.fusion, BidirectionalCrossAttentionFusion)):
            raise TypeError("the fused CAM is defined for the cross-attention hybrid")
        self.model = model
        self._out: dict[str, torch.Tensor] = {}

    def __call__(self, x: torch.Tensor, classes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        fusion = self.model.fusion
        handles = [fusion.local_norm.register_forward_hook(lambda m, a, out: self._out.__setitem__("local", out)),
                   fusion.global_norm.register_forward_hook(lambda m, a, out: self._out.__setitem__("global", out))]
        try:
            logits = self.model(x)
            local, glob = self._out["local"].float(), self._out["global"].float()  # (B, N, d)
        finally:
            for h in handles:
                h.remove()
            self._out.clear()
        with torch.autocast(device_type=x.device.type, enabled=False):  # the map in float32 under AMP
            w = self.model.head.weight.float()[classes]  # (B, 2d)
            d = local.shape[-1]
            cam = torch.einsum("bnd,bd->bn", local, w[:, :d]) + torch.einsum("bnd,bd->bn", glob, w[:, d:])
        side = int(round(cam.shape[1] ** 0.5))
        return logits, cam.view(-1, side, side)


def mask_grid(masks: torch.Tensor, side: int) -> torch.Tensor:
    """(B, H, W) 0/1 masks -> (B, side, side) lesion share of each grid cell."""
    return F.adaptive_avg_pool2d(masks.float()[:, None], side)[:, 0]


def outside_share(cam: torch.Tensor, lesion: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Mean over images of sum(ReLU(cam) * (1 - lesion)) / sum(ReLU(cam)); an all-zero map contributes 0."""
    pos = cam.clamp_min(0).flatten(1)
    return ((pos * (1 - lesion.flatten(1))).sum(dim=1) / (pos.sum(dim=1) + eps)).mean()


def consistency(cam_a: torch.Tensor, cam_b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Mean of 1 - cosine similarity of the positive parts of two maps."""
    return (1 - F.cosine_similarity(cam_a.clamp_min(0).flatten(1), cam_b.clamp_min(0).flatten(1), dim=1, eps=eps)).mean()


def add_pixel_noise(x: torch.Tensor, std: float) -> torch.Tensor:
    """Gaussian noise of `std` on the 0-1 pixel scale, added to an ImageNet-normalised batch (no clipping)."""
    scale = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return x + torch.randn_like(x) * (std / scale)


class MaskGuidedLoss(nn.Module):
    """CE (the run's loss) + lam * outside_share + mu * consistency, all on the true class."""

    def __init__(self, model: HybridEffNetSwin, base_loss: nn.Module, lam: float, mu: float, noise_std: float):
        super().__init__()
        self.cam, self.base_loss, self.lam, self.mu, self.noise_std = FusedCAM(model), base_loss, lam, mu, noise_std

    def forward(self, x: torch.Tensor, y: torch.Tensor, masks: torch.Tensor) -> tuple[torch.Tensor, dict]:
        logits, cam = self.cam(x, y)
        ce = self.base_loss(logits.float(), y)
        mask_term = outside_share(cam, mask_grid(masks, cam.shape[-1]))
        loss, parts = ce + self.lam * mask_term, {"ce": ce.item(), "mask": mask_term.item()}
        if self.mu > 0:
            _, cam_noisy = self.cam(add_pixel_noise(x, self.noise_std), y)
            cons = consistency(cam, cam_noisy)
            loss = loss + self.mu * cons
            parts["consistency"] = cons.item()
        return loss, parts
