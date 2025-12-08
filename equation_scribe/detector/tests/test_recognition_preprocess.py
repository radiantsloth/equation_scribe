import math
import pytest
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from pathlib import Path
import sys

# Our module
from equation_scribe.recognition.preprocess import (
    crop_page_image,
    deskew_crop,
    normalize_for_recognition,
    augment_for_recognition,
)


def make_test_text_image(w=200, h=100, text="E=mc^2"):
    """
    Create a synthetic page image with some text (PIL). Useful as a stand-in for rendered equations.
    """
    im = Image.new("RGB", (w, h), color=(255, 255, 255))
    d = ImageDraw.Draw(im)
    # Use default font; don't rely on external TTF
    d.rectangle([20, 20, 180, 60], outline=(0, 0, 0), width=2)
    d.text((30, 25), text, fill=(0, 0, 0))
    return im


def test_crop_and_normalize():
    page = make_test_text_image(200, 100, text="a+b=c")
    bbox = (18, 18, 182, 62)
    crop = crop_page_image(page, bbox)
    assert crop.size[0] > 0 and crop.size[1] > 0

    arr = normalize_for_recognition(crop, target_h=32, channel_first=True)
    # channel_first -> shape (3, H, W)
    assert arr.dtype == np.float32
    assert arr.shape[1] == 32
    assert arr.min() >= 0.0 and arr.max() <= 1.0


@pytest.mark.skipif(
    not pytest.importorskip("cv2", reason="opencv required for deskew test"),
    reason="opencv not available",
)
def test_deskew_estimation_and_correction():
    # Make a box and rotate it by a known angle, then ensure deskew recovers it roughly.
    base = Image.new("RGB", (300, 120), (255, 255, 255))
    d = ImageDraw.Draw(base)
    d.rectangle([60, 20, 240, 80], fill=(0, 0, 0))
    # rotate the image by a known amount
    rot_angle = 15.0
    rotated = base.rotate(rot_angle, resample=Image.BICUBIC, expand=True, fillcolor=(255,255,255))

    deskewed_img, angle = deskew_crop(rotated, return_angle=True, expand=True)
    # angle is absolute magnitude of rotation in degrees (our implementation returns abs(angle))
    assert angle is not None
    assert abs(angle - rot_angle) <= 3.0, f"expected ~{rot_angle}° got {angle}°"

    # The deskewed image should have the main rectangle axis aligned (min area rect should be near 0 deg)
    # Quick sanity: the deskewed image should be not identical to rotated and should be mostly white background
    assert deskewed_img.size[0] > 0 and deskewed_img.size[1] > 0


def test_augment_deterministic_seed():
    img = make_test_text_image(120, 40, text="test")
    a1 = augment_for_recognition(img, rotate_max_deg=5.0, add_noise_std=0.02, blur_radius=1.0, seed=123)
    a2 = augment_for_recognition(img, rotate_max_deg=5.0, add_noise_std=0.02, blur_radius=1.0, seed=123)
    # With same seed, augmentations should be identical
    assert list(a1.getdata()) == list(a2.getdata())
