"""Grad-CAM variants for the exploratory Part B trust comparison: SmoothGrad-CAM and Score-CAM.

All three methods share one interface, explainer(x, targets, seeds) -> (logits, {target: {map: (B, h, w)}}),
on the same layers and map names as gradcam.py (explain_layers): "gradcam" for ConvNeXt-T, "gradcam_local"
and "gradcam_fused" for the hybrid. The map name gives the layer; the method is given by the run's
trust directory. seeds (one per image) make any noise reproducible; the same seed is used for an image in
every phase of scripts/eval_trust.py, so the clean and the perturbed image share their SmoothGrad noise.

SmoothGrad-CAM: Grad-CAM averaged over `samples` noisy copies of the input (Smilkov et al. 2017, SmoothGrad;
not in Related_papers, cited as external). Noise std = spread * (max - min) of each input tensor; samples 50
and spread 0.15 as in the code of Adebayo et al. 2018 (Related_papers/Sanity Checks for Saliency Maps,
code/src/randomize_inceptionv3.py:78-82). The target class is resolved on the clean image, and the
post-ReLU maps are averaged.

Score-CAM (Wang et al. 2020; external): each channel k of the layer is upsampled to the image, min-max scaled
to [0, 1] and multiplied with the (normalised) input, so masked-out pixels take the dataset-mean colour, the
same fill as the deletion curves. Its weight is the softmax probability of the target class for that masked
input; map = ReLU(sum_k w_k A_k) on the layer's grid. Constant channels get weight 0. For the fused map the
channels of both fusion token streams are used together, matching the sum in gradcam.py.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from skin_lesion_xai.xai.gradcam import GradCAM, _as_grid, explain_layers

METHODS = ("gradcam", "smoothgrad", "scorecam")


def noise_seed(seed: int, sample: int) -> int:
    """Seed of one image's noise for one SmoothGrad sample (independent of the stability noise seeds)."""
    return int(np.random.SeedSequence([seed, sample, 1]).generate_state(1)[0])


def _targets(logits: torch.Tensor, classes: torch.Tensor | None) -> torch.Tensor:
    return logits.argmax(dim=1) if classes is None else classes.to(logits.device)


class GradCAMExplainer:
    def __init__(self, model: nn.Module):
        self.cam = GradCAM(model)

    def __call__(self, x: torch.Tensor, targets: dict, seeds: list[int] | None = None) -> tuple[torch.Tensor, dict]:
        return self.cam(x, targets)


class SmoothGradCAM:
    def __init__(self, model: nn.Module, samples: int = 50, spread: float = 0.15):
        self.model, self.cam, self.samples, self.spread = model, GradCAM(model), samples, spread

    def __call__(self, x: torch.Tensor, targets: dict, seeds: list[int] | None = None) -> tuple[torch.Tensor, dict]:
        if seeds is None or len(seeds) != len(x):
            raise ValueError("SmoothGrad-CAM needs one seed per image")
        self.model.eval()
        with torch.no_grad(), torch.autocast(device_type=x.device.type, enabled=False):
            logits = self.model(x.float())
        resolved = {name: _targets(logits, c) for name, c in targets.items()}
        sigma = self.spread * (x.flatten(1).amax(dim=1) - x.flatten(1).amin(dim=1))
        sums: dict[str, dict[str, np.ndarray]] = {}
        for k in range(self.samples):
            noise = torch.stack([torch.randn(x.shape[1:], device=x.device,
                                             generator=torch.Generator(x.device).manual_seed(noise_seed(s, k)))
                                 for s in seeds])  # drawn on the device: no host-to-GPU copy per sample
            _, maps = self.cam(x + sigma.view(-1, 1, 1, 1) * noise, resolved)
            for t, per_map in maps.items():
                acc = sums.setdefault(t, {})
                for m, v in per_map.items():
                    acc[m] = acc.get(m, 0) + v
        return logits.detach(), {t: {m: v / self.samples for m, v in per_map.items()} for t, per_map in sums.items()}


class ScoreCAM:
    def __init__(self, model: nn.Module, forward_batch: int = 64, amp: bool = True, maps: list[str] | None = None):
        """maps: the output maps to compute (each costs one forward pass per channel); None = all."""
        self.model, self.forward_batch, self.amp = model, forward_batch, amp
        layers, combine = explain_layers(model)
        # output map -> hooked layers whose channels it uses
        parts = {n: [n] for n, layer in layers.items() if layer.output} | dict(combine)
        if maps is not None and set(maps) - set(parts):
            raise ValueError(f"maps {sorted(set(maps) - set(parts))} not defined for {type(model).__name__}")
        self.parts = {n: p for n, p in parts.items() if maps is None or n in maps}
        self.layers = {n: layers[n] for p in self.parts.values() for n in p}
        self._acts: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def _channel_scores(self, image: torch.Tensor, channels: torch.Tensor, target: int) -> torch.Tensor:
        """(C,) softmax probability of `target` for the image masked by each upsampled, min-max scaled channel."""
        size = image.shape[-2:]
        flat = channels.flatten(1)
        lo, hi = flat.amin(dim=1), flat.amax(dim=1)
        live = (hi - lo) > 1e-12
        scores = torch.zeros(len(channels), device=image.device)
        idx = torch.nonzero(live).flatten()
        for start in range(0, len(idx), self.forward_batch):
            chunk = idx[start:start + self.forward_batch]
            up = F.interpolate(channels[chunk, None].float(), size=size, mode="bilinear", align_corners=False)
            span = (hi[chunk] - lo[chunk]).view(-1, 1, 1, 1)
            masks = ((up - lo[chunk].view(-1, 1, 1, 1)) / span).clamp_(0, 1)
            with torch.autocast(device_type=image.device.type, dtype=torch.float16,
                                enabled=self.amp and image.device.type == "cuda"):
                logits = self.model(image[None] * masks)
            scores[chunk] = logits.float().softmax(dim=1)[:, target]
        return scores

    @torch.no_grad()
    def __call__(self, x: torch.Tensor, targets: dict, seeds: list[int] | None = None) -> tuple[torch.Tensor, dict]:
        self.model.eval()

        def hook(name):
            return lambda module, args, out: self._acts.__setitem__(name, out)

        handles = [layer.module.register_forward_hook(hook(n)) for n, layer in self.layers.items()]
        try:
            with torch.autocast(device_type=x.device.type, enabled=False):
                logits = self.model(x.float())
            grids = {n: _as_grid(self._acts[n].float(), layer) for n, layer in self.layers.items()}
        finally:
            for h in handles:
                h.remove()
            self._acts.clear()
        results = {}
        for t, classes in targets.items():
            cls = _targets(logits, classes)
            out = {}
            for name, parts in self.parts.items():
                acts = torch.cat([grids[p] for p in parts], dim=1)  # (B, C, h, w)
                maps = []
                for j in range(len(x)):
                    w = self._channel_scores(x[j], acts[j], int(cls[j]))
                    maps.append((w.view(-1, 1, 1) * acts[j]).sum(dim=0).clamp_min(0))
                out[name] = torch.stack(maps).cpu().numpy()
            results[t] = out
        return logits.detach(), results


def build_explainer(method: str, model: nn.Module, maps: list[str], samples: int, spread: float,
                    forward_batch: int, amp: bool):
    if method == "gradcam":
        return GradCAMExplainer(model)
    if method == "smoothgrad":
        return SmoothGradCAM(model, samples, spread)
    if method == "scorecam":
        return ScoreCAM(model, forward_batch, amp, maps)
    raise ValueError(f"unknown explanation method {method}; expected one of {METHODS}")
