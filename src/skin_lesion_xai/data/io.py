"""Image and mask readers. Kept free of torch so multiprocessing workers stay light."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_mask(path: Path) -> np.ndarray:
    """Read a lesion mask as uint8 {0, 1}."""
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"cannot read mask: {path}")
    return (mask > 127).astype(np.uint8)
