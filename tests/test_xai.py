import numpy as np
import pytest
import timm
import torch
from torch import nn

from skin_lesion_xai.config import ModelConfig
from skin_lesion_xai.models.factory import build_model
from skin_lesion_xai.xai.attention_maps import attention_maps, key_attention, rollout
from skin_lesion_xai.xai.gradcam import CamLayer, GradCAM, freeze
from skin_lesion_xai.xai.localization import (
    bbox, bbox_iou, binarize, boundary_band, centre_gaussian, largest_component, score_map, upsample,
)

SWEEP = np.array([0.05 * i for i in range(1, 20)])


def square(size: int, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    m = np.zeros((size, size), dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


def score(heatmap, mask, pointing=True):
    return score_map(heatmap.astype(np.float32), mask, 0.15, SWEEP, boundary_band(mask, 2), pointing)


def test_perfect_map_scores_one_and_disjoint_map_scores_zero():
    mask = square(40, 10, 30, 10, 30)
    s = score(mask.astype(float), mask)
    for k in ("iou", "dice", "iou_auc", "bbox_iou", "pointing", "energy_in_mask"):
        assert s[k] == pytest.approx(1.0)
    s = score(square(40, 0, 5, 0, 5).astype(float), mask)
    for k in ("iou", "dice", "iou_auc", "bbox_iou", "pointing", "energy_in_mask"):
        assert s[k] == pytest.approx(0.0)


def test_half_overlap_has_known_iou_and_dice():
    mask = square(40, 0, 20, 0, 20)  # 400 px
    heat = square(40, 0, 20, 10, 30).astype(float)  # 400 px, 200 shared
    s = score(heat, mask)
    assert s["iou"] == pytest.approx(200 / 600)
    assert s["dice"] == pytest.approx(2 * 200 / 800)
    assert s["energy_in_mask"] == pytest.approx(0.5)


def test_threshold_is_a_fraction_of_the_max():
    heat = np.full((10, 10), 0.1)
    heat[2:5, 2:5] = 1.0
    assert binarize(heat, 0.15).sum() == 9  # 0.1 < 0.15 * 1.0
    assert binarize(heat * 0.5, 0.15).sum() == 9  # scale-free
    assert not binarize(np.zeros((4, 4)), 0.15).any()


def test_box_iou_uses_the_largest_region():
    binary = square(40, 0, 4, 0, 4) | square(40, 20, 30, 20, 30)
    assert bbox(largest_component(binary)) == (20, 20, 29, 29)
    assert bbox_iou((0, 0, 9, 9), (0, 5, 9, 14)) == pytest.approx(50 / 150)


def test_boundary_band_follows_the_border():
    mask = square(40, 10, 30, 10, 30)
    band = boundary_band(mask, 2)
    assert band[10, 20] and band[20, 10]  # on the border
    assert not band[20, 20] and not band[0, 0]  # lesion centre, far background


def test_empty_map_and_uniform_reference():
    mask = square(20, 5, 15, 5, 15)
    s = score(np.zeros((20, 20)), mask)
    assert s["map_empty"]
    for k in ("iou", "dice", "iou_auc", "bbox_iou", "pointing", "energy_in_mask", "boundary_energy"):
        assert s[k] == 0.0, k  # never NaN: an empty map must not drop out of the averages
    s = score(np.ones((20, 20)), mask, pointing=False)
    assert s["iou"] == pytest.approx(mask.mean()) and np.isnan(s["pointing"])
    g = centre_gaussian(21, 0.25)
    assert g.max() == pytest.approx(1.0) and np.unravel_index(g.argmax(), g.shape) == (10, 10)


def test_upsample_scales_each_map_to_unit_max():
    maps = np.random.default_rng(0).random((3, 7, 7)).astype(np.float32) * np.array([1, 5, 0.1])[:, None, None]
    up = upsample(maps, 224)
    assert up.shape == (3, 224, 224)
    assert np.allclose(up.reshape(3, -1).max(axis=1), 1.0) and up.min() >= 0


def test_gradcam_matches_the_grad_cam_library():
    from pytorch_grad_cam import GradCAM as LibGradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    torch.manual_seed(0)
    model = timm.create_model("resnet18", pretrained=False, num_classes=5).eval()
    x = torch.randn(2, 3, 224, 224)
    targets = torch.tensor([1, 3])
    ours = GradCAM(model, {"gradcam": CamLayer(model.layer4)})(x, {"t": targets})[1]["t"]["gradcam"]
    with LibGradCAM(model=model, target_layers=[model.layer4]) as lib:
        theirs = lib(input_tensor=x, targets=[ClassifierOutputTarget(int(t)) for t in targets])
    import cv2

    def min_max(a):
        return (a - a.min()) / (a.max() - a.min() + 1e-7)

    for o, t in zip(ours, theirs):  # the library min-max scales, resizes with OpenCV, then min-max scales again
        assert np.allclose(min_max(cv2.resize(min_max(o), (224, 224))), t, atol=1e-4)


def test_batched_gradcam_equals_one_image_at_a_time():
    torch.manual_seed(0)
    model = freeze(timm.create_model("resnet18", pretrained=False, num_classes=5).eval())
    cam = GradCAM(model, {"gradcam": CamLayer(model.layer4)})
    x = torch.randn(3, 3, 64, 64)
    batched = cam(x, {"pred": None})[1]["pred"]["gradcam"]
    single = np.concatenate([cam(x[i:i + 1], {"pred": None})[1]["pred"]["gradcam"] for i in range(3)])
    assert np.allclose(batched, single, atol=1e-5)


def test_rollout_of_uniform_attention_is_uniform():
    n, b = 1 + 16, 2
    attn = [torch.full((b, 4, n, n), 1.0 / n) for _ in range(3)]
    out = rollout(attn, prefix_tokens=1)
    assert out.shape == (b, 4, 4)
    assert torch.allclose(out, out[:, :1, :1].expand_as(out))


def test_key_attention_sums_to_one():
    attn = torch.rand(2, 8, 49, 49).softmax(dim=-1)
    assert torch.allclose(key_attention(attn).sum(dim=1), torch.ones(2))


@pytest.mark.parametrize("fusion", ["cross_attention", "concat"])
def test_hybrid_maps_have_the_7x7_grid(fusion):
    cfg = ModelConfig(name="hybrid", arch="hybrid", pretrained=False, fusion=fusion,
                      local_timm_name="tf_efficientnetv2_s.in21k_ft_in1k",
                      global_timm_name="swin_tiny_patch4_window7_224.ms_in22k_ft_in1k", match_pretrained_norm=True)
    torch.manual_seed(0)
    model = build_model(cfg, num_classes=7)
    nn.init.normal_(model.head.weight, std=0.02)  # zero-init head would give zero gradients
    model = freeze(model).eval()
    x = torch.randn(2, 3, 224, 224)
    _, maps = GradCAM(model)(x, {"pred": None})
    assert set(maps["pred"]) == {"gradcam_local", "gradcam_global", "gradcam_fused"}
    for m in maps["pred"].values():
        assert m.shape == (2, 7, 7) and (m >= 0).all() and m.any()
    attn = attention_maps(model, x)
    if fusion == "cross_attention":
        assert set(attn) == {"xattn_local_to_global", "xattn_global_to_local"}
        for m in attn.values():
            assert m.shape == (2, 7, 7) and np.allclose(m.sum(axis=(1, 2)), 1.0, atol=1e-5)
    else:
        assert attn == {}


def test_vit_gradcam_and_rollout_shapes():
    torch.manual_seed(0)
    model = freeze(build_model(ModelConfig(name="vit", arch="timm", pretrained=False,
                                           timm_name="vit_base_patch16_224.augreg2_in21k_ft_in1k"), 7).eval())
    x = torch.randn(1, 3, 224, 224)
    _, maps = GradCAM(model)(x, {"pred": None})
    assert maps["pred"]["gradcam"].shape == (1, 14, 14) and maps["pred"]["gradcam"].any()
    roll = attention_maps(model, x)["rollout"]
    assert roll.shape == (1, 14, 14) and (roll > 0).all()
    assert all(b.attn.fused_attn for b in model.blocks)  # restored after the rollout pass
