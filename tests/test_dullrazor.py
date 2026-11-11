import cv2
import numpy as np

from skin_lesion_xai.data.dullrazor import remove_hair

SKIN = (200, 160, 140)
LESION = (120, 80, 60)


def _synthetic_image():
    """Skin background, a large dark lesion, and thin black 'hairs' crossing it."""
    image = np.full((300, 400, 3), SKIN, dtype=np.uint8)
    cv2.circle(image, (200, 150), 70, LESION, -1)
    lesion = np.zeros(image.shape[:2], np.uint8)
    cv2.circle(lesion, (200, 150), 60, 1, -1)  # interior, away from the border
    hairs = np.zeros(image.shape[:2], np.uint8)
    for (x0, y0, x1, y1) in [(0, 20, 400, 280), (50, 300, 350, 0), (0, 150, 400, 140)]:
        cv2.line(image, (x0, y0), (x1, y1), (20, 15, 10), 2)
        cv2.line(hairs, (x0, y0), (x1, y1), 1, 2)
    return image, hairs.astype(bool), lesion.astype(bool)


def test_hair_detected_and_lesion_kept():
    image, hairs, lesion = _synthetic_image()
    cleaned, mask = remove_hair(image, kernel_size=17, threshold=10, return_mask=True)
    detected = mask > 0
    # the hair mask is expected to spill ~1 px past the drawn line (it covers the hair edge),
    # so "lesion away from hair" excludes a 3 px band around each hair
    near_hair = cv2.dilate(hairs.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    assert detected[hairs].mean() > 0.8, "most hair pixels should be detected"
    assert detected[lesion & ~near_hair].mean() < 0.05, "lesion interior should not be flagged as hair"
    # hair pixels get brighter after inpainting, the lesion interior is left alone
    assert cleaned[hairs].mean() > image[hairs].mean() + 50
    assert np.abs(cleaned[lesion & ~detected].astype(int) - image[lesion & ~detected]).max() == 0


def test_round_dots_are_not_hair():
    """Isolated dark dots/globules (dermoscopic structures) must survive; only the long hair is removed.

    A dot touching a hair joins the hair's connected region and is removed with it (known limitation),
    so the dots here are placed at least 10 px from every hair.
    """
    image, hairs, _ = _synthetic_image()
    dist_to_hair = cv2.distanceTransform((~hairs).astype(np.uint8), cv2.DIST_L2, 3)
    dots = np.zeros(image.shape[:2], np.uint8)
    for x, y in [(190, 100), (250, 130), (145, 160), (205, 190)]:
        assert dist_to_hair[y, x] >= 10, f"test setup: dot {(x, y)} too close to a hair"
        cv2.circle(image, (x, y), 4, (30, 20, 15), -1)
        cv2.circle(dots, (x, y), 3, 1, -1)
    _, mask = remove_hair(image, return_mask=True)
    detected = mask > 0
    assert detected[dots.astype(bool) & ~hairs].mean() < 0.05
    assert detected[hairs].mean() > 0.8


def test_shape_and_dtype_preserved():
    image, _, _ = _synthetic_image()
    cleaned = remove_hair(image)
    assert cleaned.shape == image.shape and cleaned.dtype == np.uint8
