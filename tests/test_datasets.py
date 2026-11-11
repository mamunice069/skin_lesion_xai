import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from skin_lesion_xai.config import DataConfig, relocate_data_config
from skin_lesion_xai.data.datasets import SkinLesionDataset, load_split_frame, read_mask
from skin_lesion_xai.data.samplers import balanced_sampler, class_weights
from skin_lesion_xai.data.transforms import build_eval_transform, build_train_transform


@pytest.fixture
def toy_dataset_dir(tmp_path):
    """Two 450x600 images: green background, red ellipse = lesion, mask marks the ellipse."""
    (tmp_path / "img").mkdir()
    (tmp_path / "mask").mkdir()
    for i in range(2):
        image = np.zeros((450, 600, 3), np.uint8)
        image[:] = (0, 255, 0)  # BGR green
        mask = np.zeros((450, 600), np.uint8)
        cv2.ellipse(mask, (250 + 50 * i, 220), (140, 90), 30, 0, 360, 255, -1)
        image[mask > 0] = (0, 0, 255)  # BGR red
        cv2.imwrite(str(tmp_path / "img" / f"img{i}.png"), image)
        cv2.imwrite(str(tmp_path / "mask" / f"img{i}_segmentation.png"), mask)
    frame = pd.DataFrame({"image_id": ["img0", "img1"], "label": [0, 1]})
    return tmp_path, frame


@pytest.mark.parametrize("build", [build_train_transform, build_eval_transform])
def test_image_and_mask_stay_aligned(toy_dataset_dir, build):
    root, frame = toy_dataset_dir
    ds = SkinLesionDataset(frame, root / "img", build(224), image_ext=".png",
                           mask_root=root / "mask", return_mask=True)
    torch.manual_seed(0)
    for _ in range(10):  # several random augmentations
        item = ds[0]
        image, mask = item["image"], item["mask"].bool()
        assert image.shape == (3, 224, 224) and mask.shape == (224, 224)
        assert set(item["mask"].unique().tolist()) <= {0, 1}
        if mask.sum() < 50 or (~mask).sum() < 50:
            continue
        redness = image[0] - image[1]  # normalised R - G: high on lesion, low on background
        assert redness[mask].mean() > 0 > redness[~mask].mean()


def test_missing_mask_root_rejected(toy_dataset_dir):
    root, frame = toy_dataset_dir
    with pytest.raises(ValueError):
        SkinLesionDataset(frame, root / "img", return_mask=True)


def test_class_weights_and_sampler():
    labels = np.array([0] * 90 + [1] * 10)
    w = class_weights(labels, 2)
    assert w[1] > w[0] and torch.isclose(w.mean(), torch.tensor(1.0))
    drawn = labels[list(balanced_sampler(labels, 2, seed=0))]
    assert 0.35 < drawn.mean() < 0.65  # roughly balanced after resampling


def test_kaggle_data_config_relocates_to_the_local_one():
    kaggle = DataConfig.from_yaml("configs/data/isic2020_kaggle.yaml")
    saved = vars(kaggle) | {"use_hair_removal": True}  # what a Kaggle run stores in config.json
    local = relocate_data_config(saved, "configs/data/isic2020.yaml")
    reference = DataConfig.from_yaml("configs/data/isic2020.yaml")
    assert local.raw_path(local.image_dir) == reference.raw_path(reference.image_dir)
    assert local.use_hair_removal and local.classes == kaggle.classes  # run settings are kept
    assert str(kaggle.raw_path(kaggle.image_dir)).replace("\\", "/").endswith("/kaggle/input/isic2020-256-local/images")
    with pytest.raises(ValueError, match="different data"):
        relocate_data_config(vars(kaggle), "configs/data/ham10000.yaml")


def test_real_ham10000_masks_are_binary_and_match_images():
    """Checks a sample of the real HAM10000 data; skipped until the split CSV exists."""
    cfg = DataConfig.from_yaml("configs/data/ham10000.yaml")
    try:
        frame = load_split_frame(cfg)
    except FileNotFoundError:
        pytest.skip("run scripts/prepare_ham10000.py first")
    for image_id in frame["image_id"].sample(50, random_state=0):
        raw = cv2.imread(str(cfg.mask_root / f"{image_id}{cfg.mask_suffix}"), cv2.IMREAD_GRAYSCALE)
        assert set(np.unique(raw).tolist()) <= {0, 255}, f"{image_id}: mask is not strictly 0/255"
        image = cv2.imread(str(cfg.raw_path(cfg.image_dir) / f"{image_id}{cfg.image_ext}"))
        assert read_mask(cfg.mask_root / f"{image_id}{cfg.mask_suffix}").shape == image.shape[:2]
