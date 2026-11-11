"""Forward-shape checks for every model config. pretrained=False, so no weights are downloaded."""
from pathlib import Path

import pytest
import torch

from skin_lesion_xai.config import PROJECT_DIR, ModelConfig
from skin_lesion_xai.models.backbones import FeatureBackbone
from skin_lesion_xai.models.factory import build_model, head_parameters

MODEL_CONFIGS = sorted((PROJECT_DIR / "configs" / "model").glob("*.yaml"))


@pytest.mark.parametrize("path", MODEL_CONFIGS, ids=lambda p: Path(p).stem)
def test_forward_shape_and_head(path):
    cfg = ModelConfig.from_yaml(path)
    model = build_model(cfg, num_classes=7, pretrained=False).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 7)
    head = head_parameters(model)
    assert head and sum(p.numel() for p in head) < sum(p.numel() for p in model.parameters())


@pytest.mark.parametrize("timm_name,channels", [
    ("tf_efficientnetv2_s", 1280),
    ("swin_tiny_patch4_window7_224", 768),
])
def test_backbone_returns_7x7_nchw_grid(timm_name, channels):
    backbone = FeatureBackbone(timm_name, pretrained=False).eval()
    with torch.no_grad():
        features = backbone(torch.randn(1, 3, 224, 224))
    assert features.shape == (1, channels, 7, 7)
    assert backbone.out_channels == channels


def test_cross_attention_maps_are_distributions():
    cfg = ModelConfig.from_yaml(PROJECT_DIR / "configs" / "model" / "hybrid.yaml")
    model = build_model(cfg, num_classes=7, pretrained=False).eval()
    model.fusion.set_store_attention(True)
    with torch.no_grad():
        model(torch.randn(2, 3, 224, 224))
    maps = model.fusion.attention_maps()
    assert len(maps) == cfg.fusion_layers
    for layer in maps:
        for attn in layer.values():
            assert attn.shape == (2, cfg.fusion_heads, 49, 49)
            torch.testing.assert_close(attn.sum(dim=-1), torch.ones(2, cfg.fusion_heads, 49))


def test_stored_attention_matches_fused_kernel():
    from skin_lesion_xai.models.fusion import BidirectionalCrossAttentionFusion
    fusion = BidirectionalCrossAttentionFusion(16, 8, dim=32, layers=2, heads=4).eval()
    local, global_ = torch.randn(2, 16, 7, 7), torch.randn(2, 8, 7, 7)
    with torch.no_grad():
        fast = fusion(local, global_)
        fusion.set_store_attention(True)
        explicit = fusion(local, global_)
    torch.testing.assert_close(fast, explicit, rtol=1e-4, atol=1e-5)


def test_hybrid_head_params_are_fusion_and_head_and_start_uniform():
    cfg = ModelConfig.from_yaml(PROJECT_DIR / "configs" / "model" / "hybrid.yaml")
    model = build_model(cfg, num_classes=7, pretrained=False).eval()
    head_ids = {id(p) for p in head_parameters(model)}
    assert head_ids == {id(p) for m in (model.fusion, model.head) for p in m.parameters()}
    assert not head_ids & {id(p) for p in model.local_backbone.parameters()}
    with torch.no_grad():
        probs = model(torch.randn(2, 3, 224, 224)).softmax(dim=1)
    torch.testing.assert_close(probs, torch.full((2, 7), 1 / 7))  # zero-init head


def test_input_renorm_maps_imagenet_to_target_stats_and_is_not_saved():
    from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

    from skin_lesion_xai.data.transforms import IMAGENET_MEAN, IMAGENET_STD
    from skin_lesion_xai.models.backbones import InputRenorm
    assert tuple(IMAGENET_MEAN) == tuple(IMAGENET_DEFAULT_MEAN) and tuple(IMAGENET_STD) == tuple(IMAGENET_DEFAULT_STD)
    raw = torch.rand(1, 3, 4, 4)
    imagenet = (raw - torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)) / torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    torch.testing.assert_close(InputRenorm((0.5,) * 3, (0.5,) * 3)(imagenet), (raw - 0.5) / 0.5)
    torch.testing.assert_close(InputRenorm(IMAGENET_MEAN, IMAGENET_STD)(imagenet), imagenet)
    # opt-in renorm adds no checkpoint keys, so old checkpoints load into either variant
    plain = FeatureBackbone("tf_efficientnetv2_s", pretrained=False)
    renorm = FeatureBackbone("tf_efficientnetv2_s", pretrained=False, match_pretrained_norm=True)
    assert plain.state_dict().keys() == renorm.state_dict().keys()


def test_grad_checkpointing_keeps_gradients():
    cfg = ModelConfig(name="t", arch="single_branch", timm_name="tf_efficientnetv2_s", pretrained=False)
    model = build_model(cfg, num_classes=3)
    model.set_grad_checkpointing(True)
    model.train()
    nn_head = model.head
    torch.nn.init.normal_(nn_head.weight)  # zero-init head would block gradients to the backbone
    model(torch.randn(2, 3, 224, 224)).sum().backward()
    first_conv = next(p for p in model.backbone.parameters() if p.ndim == 4)
    assert first_conv.grad is not None and first_conv.grad.abs().sum() > 0
