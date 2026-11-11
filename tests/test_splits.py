import numpy as np
import pandas as pd
import pytest

from skin_lesion_xai.config import DataConfig
from skin_lesion_xai.data.splits import assert_no_group_leakage, grouped_split


def _synthetic_frame(seed: int = 0) -> pd.DataFrame:
    """600 groups with 1-5 images each and imbalanced labels (like HAM10000 lesions)."""
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(600):
        label = rng.choice(3, p=[0.7, 0.2, 0.1])
        rows += [{"group_id": f"g{g}", "label": label}] * rng.integers(1, 6)
    return pd.DataFrame(rows)


def test_split_has_no_group_leakage_and_close_ratios():
    frame = _synthetic_frame()
    frame["split"] = grouped_split(frame["label"], frame["group_id"], {"train": 0.7, "val": 0.15, "test": 0.15})
    assert_no_group_leakage(frame)
    share = frame["split"].value_counts(normalize=True)
    assert share["train"] == pytest.approx(0.70, abs=0.05)
    assert share["val"] == pytest.approx(0.15, abs=0.04)
    assert share["test"] == pytest.approx(0.15, abs=0.04)
    for split in ("train", "val", "test"):
        assert set(frame.loc[frame["split"] == split, "label"]) == {0, 1, 2}


def test_split_is_reproducible():
    frame = _synthetic_frame()
    ratios = {"train": 0.7, "val": 0.15, "test": 0.15}
    a = grouped_split(frame["label"], frame["group_id"], ratios, seed=1)
    b = grouped_split(frame["label"], frame["group_id"], ratios, seed=1)
    assert (a == b).all()


def test_bad_ratios_rejected():
    frame = _synthetic_frame()
    with pytest.raises(ValueError):
        grouped_split(frame["label"], frame["group_id"], {"train": 0.7, "val": 0.12, "test": 0.18})


def test_leakage_detected():
    frame = pd.DataFrame({"group_id": ["a", "a", "b"], "split": ["train", "test", "val"]})
    with pytest.raises(AssertionError):
        assert_no_group_leakage(frame)


@pytest.mark.parametrize("config", ["configs/data/ham10000.yaml", "configs/data/isic2020.yaml"])
def test_real_split_csv(config):
    """Runs only after scripts/prepare_*.py has written the split CSV."""
    cfg = DataConfig.from_yaml(config)
    path = cfg.project_path(cfg.split_csv)
    if not path.exists():
        pytest.skip(f"{path} not generated yet")
    frame = pd.read_csv(path)
    assert_no_group_leakage(frame)
    assert set(frame["split"]) == {"train", "val", "test"}
    assert frame["image_id"].is_unique
    assert set(frame["label"]) == set(range(cfg.num_classes))
