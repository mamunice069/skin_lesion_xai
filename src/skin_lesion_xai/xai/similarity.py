"""Similarity of two explanation maps, as in the model randomisation test of Adebayo et al. 2018
(Related_papers/Sanity Checks for Saliency Maps, code/notebooks/inceptionv3_cascading_randomization.ipynb):
Spearman rank correlation, SSIM with Gaussian weights, and Spearman correlation of HOG features
(pixels_per_cell 16). Maps are compared at image resolution, each scaled to [0, 1] by its max (Grad-CAM is
non-negative after its ReLU, so no absolute value is needed).

A constant map has no ranking: two constant maps count as identical (1) and a constant map against a
non-constant one as unrelated (0), so all-zero maps never drop out of the averages.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from skimage.feature import hog
from skimage.metrics import structural_similarity


def _constant(v: np.ndarray) -> bool:
    return bool(np.ptp(v) == 0)


def _rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel(), b.ravel()
    ca, cb = _constant(a), _constant(b)
    if ca or cb:
        return 1.0 if ca and cb else 0.0
    return float(spearmanr(a, b)[0])


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return _rank_corr(a, b)


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    return float(structural_similarity(a, b, gaussian_weights=True, data_range=1.0))


def hog_spearman(a: np.ndarray, b: np.ndarray, pixels_per_cell: int = 16) -> float:
    cell = (pixels_per_cell, pixels_per_cell)
    return _rank_corr(hog(a, pixels_per_cell=cell), hog(b, pixels_per_cell=cell))


def similarities(a: np.ndarray, b: np.ndarray, pixels_per_cell: int = 16, with_hog: bool = True) -> dict:
    out = {"spearman": spearman(a, b), "ssim": ssim(a, b)}
    if with_hog:
        out["hog"] = hog_spearman(a, b, pixels_per_cell)
    return out
