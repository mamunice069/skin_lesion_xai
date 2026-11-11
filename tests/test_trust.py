"""Explanation-trust metrics (M4) on inputs whose answers are known."""
import importlib.util
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from skin_lesion_xai.config import PROJECT_DIR, ModelConfig
from skin_lesion_xai.eval.metrics import holm, paired_bootstrap_per_image
from skin_lesion_xai.models.factory import build_model
from skin_lesion_xai.xai.cam_variants import GradCAMExplainer, ScoreCAM, SmoothGradCAM
from skin_lesion_xai.xai.mask_guidance import FusedCAM, consistency, mask_grid, outside_share
from skin_lesion_xai.xai.faithfulness import curve_auc, perturbation_curves, pixel_order
from skin_lesion_xai.xai.randomization import cascade, randomize_step
from skin_lesion_xai.xai.similarity import similarities, spearman, ssim
from skin_lesion_xai.xai.stability import perturb, restore_map


class RegionModel(nn.Module):
    """Two logits whose gap is the mean of the image inside a fixed region: only that region matters."""

    def __init__(self, region: torch.Tensor):
        super().__init__()
        self.register_buffer("region", region.float())

    def forward(self, x):
        s = (x.mean(dim=1) * self.region).sum(dim=(1, 2)) / self.region.sum()
        return torch.stack([5 * s, -5 * s], dim=1)


def test_pixel_order_ranks_highest_first():
    rank = pixel_order(np.array([[[0.1, 0.9], [0.5, 0.5]]]))
    assert rank.tolist() == [[3, 0, 1, 2]]  # ties keep pixel order


def test_deletion_and_insertion_reward_the_map_that_points_at_the_evidence():
    size = 16
    region = torch.zeros(size, size)
    region[4:8, 4:8] = 1
    model = RegionModel(region).eval()
    x = (2 * region - 1).expand(2, 3, size, size).clone()  # +1 in the region, -1 outside: class 0, confident
    good = np.broadcast_to(region.numpy(), (2, size, size))
    bad = 1 - good
    cls = torch.zeros(2, dtype=torch.long)
    run = lambda m, mode: curve_auc(perturbation_curves(model, x, m, cls, 16, mode, blur_kernel=9, blur_sigma=3.0,
                                                        forward_batch=7, amp=False))
    assert (run(good, "deletion") < run(bad, "deletion")).all()
    assert (run(good, "insertion") > run(bad, "insertion")).all()
    curves = perturbation_curves(model, x, good, cls, 16, "deletion", forward_batch=5, amp=False)
    assert curves.shape == (2, 17) and curves[0, 0] > 0.99 and curves[0, -1] == pytest.approx(0.5)
    assert curve_auc(np.ones((1, 11)))[0] == pytest.approx(1.0)


def test_similarity_of_identical_and_constant_maps():
    rng = np.random.default_rng(0)
    a = rng.random((64, 64)).astype(np.float32)
    s = similarities(a, a.copy(), pixels_per_cell=16)
    assert s["spearman"] == pytest.approx(1) and s["ssim"] == pytest.approx(1) and s["hog"] == pytest.approx(1)
    zero = np.zeros_like(a)
    assert spearman(zero, zero) == 1.0 and spearman(a, zero) == 0.0
    assert ssim(a, 1 - a) < 0.5


def test_flip_is_undone_and_perturbations_are_reproducible():
    m = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    assert np.array_equal(restore_map(m[..., ::-1], "hflip"), m)
    x = torch.randn(2, 3, 8, 8) * 0.5
    assert torch.equal(perturb(x, "hflip"), x.flip(-1))
    n1, n2 = perturb(x, "noise_0.03", seeds=[1, 2]), perturb(x, "noise_0.03", seeds=[1, 2])
    assert torch.equal(n1, n2) and not torch.equal(n1, perturb(x, "noise_0.03", seeds=[3, 4]))
    from skin_lesion_xai.data.transforms import IMAGENET_MEAN, IMAGENET_STD
    mean, std = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    pixels = torch.rand(2, 3, 8, 8) * 0.5  # brightness 1.1 stays inside [0, 1]
    bright = perturb((pixels - mean) / std, "brightness_1.1") * std + mean
    assert torch.allclose(bright, pixels * 1.1, atol=1e-5)


@pytest.mark.parametrize("name", ["convnext_t", "hybrid"])
def test_cascade_covers_every_parameter_and_randomises_it(name):
    cfg = ModelConfig.from_yaml(PROJECT_DIR / "configs" / "model" / f"{name}.yaml")
    torch.manual_seed(0)
    model = build_model(cfg, num_classes=7, pretrained=False)
    for p in model.parameters():  # stand-in for training: move every weight away from its initial value
        p.data.add_(0.1)
    steps = cascade(model)
    covered = {id(p) for _, paths in steps for path in paths for p in model.get_submodule(path).parameters()}
    assert covered == {id(p) for p in model.parameters()}
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    bn_stats = {n: b.clone() for n, b in model.named_buffers() if "running_mean" in n}
    for _, paths in steps:
        randomize_step(model, paths)
    changed = [n for n, p in model.named_parameters() if not torch.equal(p, before[n])]
    assert len(changed) == len(before)
    assert all(torch.equal(b, bn_stats[n]) for n, b in model.named_buffers() if n in bn_stats)


def test_paired_bootstrap_per_image_and_holm():
    rng = np.random.default_rng(0)
    a = rng.random(200)
    same = paired_bootstrap_per_image(a, a, n_boot=100)
    assert same["diff"] == 0 and same["p_two_sided"] == 1.0
    better = paired_bootstrap_per_image(a, a + 0.1 + rng.normal(0, 0.01, 200), n_boot=200)
    assert better["diff"] == pytest.approx(0.1, abs=0.01) and better["diff_ci"]["low"] > 0
    correct = np.array([True] * 150 + [False] * 50)
    v = np.where(correct, 0.6, 0.4)
    gap = paired_bootstrap_per_image(v, v, n_boot=50, correct_a=correct, correct_b=correct)
    assert gap["a"] == pytest.approx(0.2) and gap["diff"] == pytest.approx(0.0)
    assert holm({"x": 0.01, "y": 0.04, "z": 0.03}) == pytest.approx({"x": 0.03, "y": 0.06, "z": 0.06})


def test_compare_xai_paired_refuses_other_image_orders(tmp_path):
    def write(run, order):
        d = run / "trust_val"
        d.mkdir(parents=True)
        rows = [{"image_id": f"img{i}", "label": i % 2, "correct": True, "map": m, "deletion_auc": 0.3,
                 "insertion_auc": 0.6} for i in order for m in ("gradcam", "gradcam_local", "ref_random")]
        pd.DataFrame(rows).to_csv(d / "per_image.csv", index=False)

    write(tmp_path / "ham10000_a", range(6))
    write(tmp_path / "ham10000_b", reversed(range(6)))
    env = dict(os.environ, PYTHONPATH=str(PROJECT_DIR / "src"))
    r = subprocess.run([sys.executable, "scripts/compare_xai_paired.py", "--runs", str(tmp_path / "ham10000_a"),
                        str(tmp_path / "ham10000_b"), "--split", "val", "--maps-b", "gradcam_local"],
                       cwd=PROJECT_DIR, env=env, capture_output=True, text=True)
    assert r.returncode != 0 and "same order" in r.stderr


def test_smoothgrad_without_noise_is_gradcam_and_scorecam_weights_the_evidence():
    torch.manual_seed(0)
    model = build_model(ModelConfig.from_yaml(PROJECT_DIR / "configs" / "model" / "convnext_t.yaml"), 7,
                        pretrained=False).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    x = torch.randn(2, 3, 64, 64)
    _, plain = GradCAMExplainer(model)(x, {"pred": None})
    _, smooth = SmoothGradCAM(model, samples=1, spread=0.0)(x, {"pred": None}, seeds=[1, 2])
    np.testing.assert_allclose(smooth["pred"]["gradcam"], plain["pred"]["gradcam"], rtol=1e-5, atol=1e-6)
    _, noisy_a = SmoothGradCAM(model, samples=2, spread=0.15)(x, {"pred": None}, seeds=[1, 2])
    _, noisy_b = SmoothGradCAM(model, samples=2, spread=0.15)(x, {"pred": None}, seeds=[1, 2])
    np.testing.assert_array_equal(noisy_a["pred"]["gradcam"], noisy_b["pred"]["gradcam"])  # seeded noise

    region = torch.zeros(16, 16, dtype=torch.bool)
    region[:8, :8] = True
    score = ScoreCAM.__new__(ScoreCAM)  # only the masking step, on a model where one region decides
    score.model, score.forward_batch, score.amp = RegionModel(region), 8, False
    channels = torch.stack([region.float(), (~region).float(), torch.ones(16, 16)])  # image-sized: no blur
    w = score._channel_scores(torch.ones(3, 16, 16), channels, target=0)
    assert w[0] > 0.99 and abs(w[1] - 0.5) < 1e-6 and w[2] == 0  # evidence kept / removed / constant channel


def test_fused_cam_equals_fused_gradcam_and_mask_losses_by_hand():
    torch.manual_seed(0)
    model = build_model(ModelConfig.from_yaml(PROJECT_DIR / "configs" / "model" / "hybrid.yaml"), 7,
                        pretrained=False).eval()
    nn.init.normal_(model.head.weight, std=0.1)  # zero-initialised head: give the classes something to say
    x, y = torch.randn(2, 3, 224, 224), torch.tensor([1, 4])
    with torch.no_grad():
        _, cam = FusedCAM(model)(x, y)
    frozen = model
    for p in frozen.parameters():
        p.requires_grad_(False)
    _, grad = GradCAMExplainer(frozen)(x, {"true": y})
    tokens = cam.shape[1] * cam.shape[2]
    np.testing.assert_allclose(cam.clamp_min(0).numpy() / tokens, grad["true"]["gradcam_fused"], rtol=1e-3, atol=1e-6)

    m = torch.tensor([[[1.0, 3.0], [0.0, -2.0]]])  # positive part 1 and 3
    lesion = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])  # only the first cell is lesion
    assert outside_share(m, lesion).item() == pytest.approx(3 / 4)
    assert consistency(m, 2 * m).item() == pytest.approx(0.0, abs=1e-6)  # same direction, other scale
    assert mask_grid(torch.ones(1, 224, 224), 7).shape == (1, 7, 7)


def load_compare_module():
    spec = importlib.util.spec_from_file_location("compare_xai_paired", PROJECT_DIR / "scripts" / "compare_xai_paired.py")
    compare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compare)
    return compare


def test_compare_aligns_a_subset_run_and_refuses_disjoint_images():
    compare = load_compare_module()
    full = pd.DataFrame({"image_id": [f"img{i}" for i in range(6)], "label": [0, 1, 0, 1, 0, 1], "v": range(6)})
    sub = full.iloc[[4, 1, 3]].reset_index(drop=True)  # a shuffled subset, as a stratified subset run
    a, b, dropped = compare.align(full, sub)
    assert a["image_id"].tolist() == ["img4", "img1", "img3"] and a["v"].tolist() == [4, 1, 3] and dropped == 3
    other = sub.assign(image_id=["x1", "x2", "x3"])
    with pytest.raises(SystemExit, match="neither is a subset"):
        compare.align(full.iloc[:3], other)


def test_compare_table_scores_randomisation_by_absolute_correlation():
    compare = load_compare_module()
    trust = pd.DataFrame([{"image_id": f"img{i}", "map": m, "deletion_auc": 0.3, "insertion_auc": 0.6,
                           "rand_full_spearman": rho, "rand_full_hog": -rho}
                          for i, rho in enumerate([-0.5, 0.2]) for m in ("gradcam", "ref_random")])
    t = compare.table(trust, None, "gradcam")
    assert t["rand_full_abs_spearman"].tolist() == [0.5, 0.2] and t["rand_full_abs_hog"].tolist() == [0.5, 0.2]
    assert "rand_full_spearman" not in compare.DIRECTION and compare.DIRECTION["rand_full_abs_hog"] == "lower"
