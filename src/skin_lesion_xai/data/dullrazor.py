"""Hair removal for dermoscopic images.

Simplified OpenCV version of DullRazor (Lee et al., 1997, "DullRazor: A software approach
to hair removal from images"), as cited in the proposal (slide 7). The original uses
directional grayscale closing, a thin-structure check and bilinear interpolation; here:
  1. black-hat morphology (closing - image) highlights thin dark structures (hair),
  2. a fixed threshold turns that into a candidate mask,
  3. thin-structure check: only connected regions that are long and thin are kept as hair;
     small or round regions (dots, globules, pigment-network fragments) are dropped,
  4. masked pixels are filled with Telea inpainting.
Lesions wider than the structuring element are not picked up by the black-hat step,
so kernel_size must be larger than a hair and smaller than the lesion.
"""
from __future__ import annotations

import cv2
import numpy as np


def keep_hair_like(mask: np.ndarray, min_length: float, min_elongation: float) -> np.ndarray:
    """Keep connected regions that look like hair.

    Length = diagonal of the region's bounding box (for a curved hair, roughly its chord).
    Elongation = length**2 / area, i.e. length / mean thickness; a straight line of
    width w scores length / w, a filled disc scores about 2.5.
    A dot that touches a hair joins the hair's region and is removed with it.
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    length = np.hypot(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT])
    area = stats[:, cv2.CC_STAT_AREA].astype(np.float64)
    keep = (length >= min_length) & (length**2 / area >= min_elongation)
    keep[0] = False  # background label
    return np.where(keep[labels], 255, 0).astype(np.uint8)


def hair_mask(
    image_rgb: np.ndarray,
    kernel_size: int = 17,
    threshold: int = 12,
    min_length_frac: float = 0.08,
    min_elongation: float = 8.0,
) -> np.ndarray:
    """Return a uint8 mask (0/255) of detected hair pixels.

    min_length_frac is relative to the image's shorter side, so the same value works at
    600x450 and 256x256. Set min_length_frac=0 and min_elongation=0 to disable the shape check.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    blackhat = cv2.GaussianBlur(blackhat, (3, 3), 0)
    _, mask = cv2.threshold(blackhat, threshold, 255, cv2.THRESH_BINARY)
    min_length = min_length_frac * min(gray.shape)
    return keep_hair_like(mask, min_length, min_elongation)


def remove_hair(
    image_rgb: np.ndarray,
    kernel_size: int = 17,
    threshold: int = 12,
    min_length_frac: float = 0.08,
    min_elongation: float = 8.0,
    dilate: int = 0,
    inpaint_radius: int = 3,
    return_mask: bool = False,
):
    """Remove hair from an RGB uint8 image. Optionally also return the hair mask.

    dilate > 0 grows the hair mask by that many pixels before inpainting, so the dark
    fringe along each hair is filled too.
    """
    if image_rgb.dtype != np.uint8 or image_rgb.ndim != 3:
        raise ValueError(f"expected HxWx3 uint8 image, got {image_rgb.dtype} {image_rgb.shape}")
    mask = hair_mask(image_rgb, kernel_size, threshold, min_length_frac, min_elongation)
    if dilate > 0:
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1)))
    cleaned = cv2.inpaint(image_rgb, mask, inpaint_radius, cv2.INPAINT_TELEA)
    return (cleaned, mask) if return_mask else cleaned
