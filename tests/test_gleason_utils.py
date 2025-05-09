from __future__ import annotations

from typing import Tuple

import numpy as np
import pytest

# -----------------------------------------------------------------------------
from gleasonxai.gleason_utils import tissue_filter_image

# -----------------------------------------------------------------------------
# Helper utilities for building synthetic test images

def make_uniform_image(rgb: Tuple[int, int, int], shape: Tuple[int, int] = (64, 64)) -> np.ndarray:  # noqa: D401
    """Return an *H×W×3* uint8 image filled with the given RGB colour."""
    return np.full((*shape, 3), rgb, dtype=np.uint8)


def make_halved_image(left_rgb: Tuple[int, int, int], right_rgb: Tuple[int, int, int], *, width: int = 64, height: int = 64) -> np.ndarray:  # noqa: D401,E501
    """Return an image whose left half is *left_rgb* and right half *right_rgb*."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    mid = width // 2
    img[:, :mid] = left_rgb
    img[:, mid:] = right_rgb
    return img


def fraction_of_image(mask: np.ndarray) -> float:
    """Return the fraction of *True* pixels in a boolean mask."""
    return mask.mean().item()


# -----------------------------------------------------------------------------
# Fixtures
@pytest.fixture(scope="module")
def rng() -> np.random.Generator:  # noqa: D401
    return np.random.default_rng(seed=2025)


# -----------------------------------------------------------------------------
# 1. Basic / sanity checks

def test_black_image_yields_empty_mask():
    img = make_uniform_image((0, 0, 0), shape=(32, 32))
    mask = tissue_filter_image(img)
    assert mask.dtype == bool
    assert mask.shape == img.shape[:2]
    assert mask.sum() == 0, "Completely black image should contain no tissue"


def test_gray_image_yields_empty_mask():
    img = make_uniform_image((128, 128, 128), shape=(40, 25))
    mask = tissue_filter_image(img)
    assert mask.sum() == 0, "Low‑saturation grey image should contain no tissue"


# -----------------------------------------------------------------------------
# 2. Detection capability on a simple synthetic image
@pytest.mark.parametrize("kernel_size,iterations", [(3, 1), (17, 2)])
def test_half_red_half_white_detection(kernel_size: int, iterations: int):
    red = (200, 0, 0)  # high saturation (tissue)
    white = (255, 255, 255)  # zero saturation (background)
    img = make_halved_image(red, white)

    mask = tissue_filter_image(img, kernel_size=kernel_size, iterations=iterations)

    # Left half should be tissue, right half background.
    h, w = img.shape[:2]
    expected_fraction = 0.5  # 50 % of the image is red
    frac = fraction_of_image(mask)
    assert pytest.approx(frac, rel=0.05) == expected_fraction, "Mask should cover roughly half the pixels (±5 %)"

    # Spot‑check individual pixels to be sure the halves are treated differently
    assert mask[0,0] == True  # left/red
    assert mask[-1,-1] == False  # right/white


# -----------------------------------------------------------------------------
# 4. The `size` parameter rescales the output mask

def test_size_parameter_downsamples_mask():
    img = make_halved_image((200, 0, 0), (255, 255, 255), width=120, height=80)
    out_size = (30, 20)  # (W,H) expected by cv2.resize in impl.
    mask = tissue_filter_image(img, size=out_size)

    assert mask.shape == out_size[::-1], "Requested output size should be respected"
    # Rough content check – still half tissue
    expected_fraction = 0.5
    assert pytest.approx(fraction_of_image(mask), rel=0.05) == expected_fraction


# -----------------------------------------------------------------------------
# 5. Robustness against extremely small inputs & large kernels

@pytest.mark.parametrize("shape", [(5, 5), (2, 8), (8, 2)])
def test_tiny_images_do_not_crash(shape):
    img = make_uniform_image((180, 0, 0), shape=shape)
    # Use a kernel larger than the image to stress the code‑path
    mask = tissue_filter_image(img, kernel_size=17, iterations=1)
    assert mask.shape == shape
    # Entire tiny image is saturated => all tissue
    assert mask.all()


# -----------------------------------------------------------------------------
# 6. Flood‑fill background logic – a hollow rectangle with a hole in the centre

def test_flood_fill_background_detection():
    # Create an image: red border, white centre – visually like a frame
    img = make_uniform_image((255, 255, 255), shape=(64, 64))
    img[10:40, 10:40] = (255, 0, 0)  # inner red square (tissue)

    # Expect tissue only in the inner square (roughly (44×44) px)
    mask = tissue_filter_image(img, kernel_size=1, open=False, close=False, flood=True)

    expected_pixels = 30 * 30
    #There some stuff with the median filter
    assert pytest.approx(expected_pixels, rel=0.1) == mask.sum(), "Mask should cover roughly 20×20 pixels (±10 %)"
    # Corners (background) should be False
    assert mask[0, 0]  == False and mask[-1, -1] == False
    # Middle of square should be tissue
    assert mask[25,25] == True


# -----------------------------------------------------------------------------
# 7. Idempotency – running twice on the same image should yield identical masks

def test_idempotent(rng):
    # Generate a random colourful image (guaranteed saturated)
    img = (rng.random((50, 70, 3)) * 255).astype(np.uint8)
    mask1 = tissue_filter_image(img)
    mask2 = tissue_filter_image(img)
    assert np.array_equal(mask1, mask2)
