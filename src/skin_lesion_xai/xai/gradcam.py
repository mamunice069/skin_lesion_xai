"""Grad-CAM (Selvaraju et al. 2017, §3, Eq. 1-2) for every model in the project.

For a class c and a feature map A (C x h x w): alpha_k = mean over locations of d y^c / d A^k, where y^c is
the pre-softmax score; map = ReLU(sum_k alpha_k A^k). The hook-and-pool structure follows the Grad-CAM
reference code (misc/utils.lua, grad_cam) and the Mahesh et al. explainer (code/gradcam_explainer.py), with
two changes from the latter: the map is not min-shifted (localization.py scales by the max, so the 15%
threshold of Grad-CAM §4.1 keeps its meaning), and the layer is the backbone output after its final
activation rather than the last Conv2d (for EfficientNetV2 that conv is followed by BN + activation).

Layers explained (explain_layers):
  - ResNet-50: layer4; ConvNeXt-T: stages[-1]; EfficientNetV2-S / Swin-T single-branch: backbone output.
    All 7x7 at 224 px.
  - ViT-B/16: patch tokens entering the last block's attention (blocks[-1].norm1), 14x14. The head reads
    the class token, so the last block's own output patch tokens carry no gradient.
  - Hybrid: "gradcam_local" (EfficientNetV2 grid), "gradcam_global" (Swin grid), and "gradcam_fused":
      cross-attention: the two normalised output token streams of the fusion, whose means feed the linear
      head, so the logit is a sum of the two streams' contributions; the fused map is
      ReLU(raw_local + raw_global) of the pre-ReLU maps (both 7x7 on the same locations);
      concat: the same sum over the two branch grids, which feed the head through average pooling.
Gradients are taken in float32 with the model's parameters frozen; the hooked outputs are made to require
grad, so autograd only records the graph above them. Samples do not interact in eval mode, so one backward
of the summed target scores gives each image its own gradients.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from skin_lesion_xai.models.backbones import SingleBranchClassifier
from skin_lesion_xai.models.fusion import BidirectionalCrossAttentionFusion, ConcatFusion
from skin_lesion_xai.models.hybrid import HybridEffNetSwin


@dataclass
class CamLayer:
    module: nn.Module
    layout: str = "nchw"  # "nchw" grid or "tokens" (B, N, C)
    prefix_tokens: int = 0  # class/register tokens dropped before reshaping tokens to a grid
    output: bool = True  # False: only used inside a combined map


def explain_layers(model: nn.Module) -> tuple[dict[str, CamLayer], dict[str, list[str]]]:
    """Layers to hook, and combined maps {name: [layer names whose pre-ReLU maps are summed]}."""
    from timm.models.convnext import ConvNeXt
    from timm.models.resnet import ResNet
    from timm.models.vision_transformer import VisionTransformer

    if isinstance(model, HybridEffNetSwin):
        layers = {"gradcam_local": CamLayer(model.local_backbone), "gradcam_global": CamLayer(model.global_backbone)}
        if isinstance(model.fusion, BidirectionalCrossAttentionFusion):
            layers["fused_local"] = CamLayer(model.fusion.local_norm, "tokens", output=False)
            layers["fused_global"] = CamLayer(model.fusion.global_norm, "tokens", output=False)
            return layers, {"gradcam_fused": ["fused_local", "fused_global"]}
        if isinstance(model.fusion, ConcatFusion):
            return layers, {"gradcam_fused": ["gradcam_local", "gradcam_global"]}
        raise TypeError(f"no Grad-CAM layers for fusion {type(model.fusion).__name__}")
    if isinstance(model, SingleBranchClassifier):
        return {"gradcam": CamLayer(model.backbone)}, {}
    if isinstance(model, VisionTransformer):
        return {"gradcam": CamLayer(model.blocks[-1].norm1, "tokens", model.num_prefix_tokens)}, {}
    if isinstance(model, ResNet):
        return {"gradcam": CamLayer(model.layer4)}, {}
    if isinstance(model, ConvNeXt):
        return {"gradcam": CamLayer(model.stages[-1])}, {}
    raise TypeError(f"no Grad-CAM layers defined for {type(model).__name__}")


def _as_grid(t: torch.Tensor, layer: CamLayer) -> torch.Tensor:
    """(B, C, h, w) view of a hooked output."""
    if layer.layout == "nchw":
        return t
    tokens = t[:, layer.prefix_tokens:]
    side = int(round(tokens.shape[1] ** 0.5))
    if side * side != tokens.shape[1]:
        raise ValueError(f"{tokens.shape[1]} tokens do not form a square grid")
    return tokens.transpose(1, 2).reshape(t.shape[0], t.shape[2], side, side)


class GradCAM:
    def __init__(self, model: nn.Module, layers: dict[str, CamLayer] | None = None,
                 combine: dict[str, list[str]] | None = None):
        if layers is None:
            layers, combine = explain_layers(model)
        self.model, self.layers, self.combine = model, layers, combine or {}
        self._acts: dict[str, torch.Tensor] = {}
        self._grads: dict[str, torch.Tensor] = {}

    def _hook(self, name: str):
        def forward_hook(module, args, out):
            if out.grad_fn is None:  # nothing below requires grad: start the graph here
                out.requires_grad_(True)
            self._acts[name] = out
            out.register_hook(lambda grad: self._grads.__setitem__(name, grad))
        return forward_hook

    def __call__(self, x: torch.Tensor, targets: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict]:
        """targets: {target name: (B,) class indices, or None for the predicted class}.
        Returns logits (B, K) and {target name: {map name: (B, h, w) float32 numpy, after ReLU}}."""
        self.model.eval()
        handles = [layer.module.register_forward_hook(self._hook(name)) for name, layer in self.layers.items()]
        try:
            with torch.autocast(device_type=x.device.type, enabled=False):
                logits = self.model(x.float())
            results = {}
            names = list(targets)
            for i, target_name in enumerate(names):
                classes = targets[target_name]
                classes = logits.argmax(dim=1) if classes is None else classes.to(logits.device)
                self._grads.clear()
                logits.gather(1, classes[:, None]).sum().backward(retain_graph=i < len(names) - 1)
                raw = {}
                for name, layer in self.layers.items():
                    act = _as_grid(self._acts[name].detach(), layer)
                    grad = _as_grid(self._grads[name], layer)
                    raw[name] = (grad.mean(dim=(2, 3), keepdim=True) * act).sum(dim=1)
                maps = {name: raw[name] for name, layer in self.layers.items() if layer.output}
                maps |= {name: sum(raw[part] for part in parts) for name, parts in self.combine.items()}
                results[target_name] = {name: m.clamp_min(0).float().cpu().numpy() for name, m in maps.items()}
        finally:
            for h in handles:
                h.remove()
            self._acts.clear()
            self._grads.clear()
        return logits.detach(), results


def freeze(model: nn.Module) -> nn.Module:
    """No parameter gradients: Grad-CAM only needs gradients of the hooked feature maps."""
    for p in model.parameters():
        p.requires_grad_(False)
    return model
