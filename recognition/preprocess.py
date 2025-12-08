"""
Recognition preprocessing utilities.

Provides:
- crop_page_image: crop a page image by bbox (x0,y0,x1,y1) with safe clipping.
- deskew_crop: compute deskew angle using OpenCV minAreaRect and return deskewed image.
- normalize_for_recognition: resize/cast/normalize crop to a target height (preserve aspect).
- augment_for_recognition: simple augment pipeline for recognition training (rotation, blur, noise).

Notes
-----
This module uses Pillow and optionally OpenCV (cv2) for robust deskewing.
If cv2 is not available, deskew_crop will raise ImportError. Augmentations are
kept simple and deterministic when a seed is provided.
"""
from __future__ import annotations

from typing import Tuple, Optional
import logging
import math
import random

from PIL import Image, ImageFilter, ImageOps
import numpy as np

logger = logging.getLogger(__name__)


def crop_page_image(page_img: Image.Image, bbox: Tuple[int, int, int, int]) -> Image.Image:
    """
    Safely crop a page image given bbox=(x0, y0, x1, y1). Clips bbox to image bounds.

    Returns a PIL Image (RGB).
    """
    if page_img.mode not in ("RGB", "RGBA", "L"):
        page_img = page_img.convert("RGB")

    x0, y0, x1, y1 = bbox
    w, h = page_img.size
    x0 = max(0, int(round(x0)))
    y0 = max(0, int(round(y0)))
    x1 = min(w, int(round(x1)))
    y1 = min(h, int(round(y1)))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Invalid bbox after clipping: {(x0,y0,x1,y1)}")

    crop = page_img.crop((x0, y0, x1, y1))
    if crop.mode == "RGBA":
        # Composite alpha against white to get RGB
        bg = Image.new("RGB", crop.size, (255, 255, 255))
        bg.paste(crop, mask=crop.split()[3])
        crop = bg
    elif crop.mode != "RGB":
        crop = crop.convert("RGB")
    return crop


def deskew_crop(
    crop_img: Image.Image, *,
    return_angle: bool = True,
    expand: bool = True,
) -> Tuple[Image.Image, Optional[float]]:
    """
    Deskew a crop by estimating its rotation angle using OpenCV's minAreaRect.

    Returns (deskewed_image, angle_in_degrees). Angle is positive if rotated clockwise
    to produce an upright image (i.e., you should rotate image by -angle to deskew — but
    this function already applies that rotation).

    Requires 'cv2' to be installed. If cv2 is missing, ImportError is raised.
    The function converts the image to gray, thresholds, finds the largest non-background
    contour area, computes minAreaRect, and derives the angle.

    expand: if True, uses PIL.rotate(expand=True) to avoid cropping after rotation.
    """
    try:
        import cv2  # imported locally to avoid failing module import if not installed
    except Exception as e:
        raise ImportError("deskew_crop requires opencv (cv2). Install opencv-python.") from e

    # Convert PIL -> numpy BGR for opencv; we only need gray
    arr = np.array(crop_img.convert("L"))
    # threshold (Otsu)
    _, th = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # find coords of non-zero pixels
    coords = cv2.findNonZero(th)
    if coords is None:
        # Nothing to deskew (blank crop)
        return (crop_img.copy(), 0.0) if return_angle else (crop_img.copy(), None)

    # Fit a min area rect to the set of points
    rect = cv2.minAreaRect(coords)
    ((cx, cy), (width, height), angle) = rect  # angle in degrees: (-90, 0]
    # Convert angle to a "rotation" that will deskew: cv2 angle semantics are awkward
    if width < height:
        angle = angle + 90.0

    # angle now is the angle to rotate the image to make rect axis-aligned.
    # PIL.rotate rotates counter-clockwise for positive angles, but our angle is how much the box
    # is rotated from horizontal. We want to rotate by -angle to deskew (clockwise).
    deskew_angle = -angle

    pil_rot = crop_img.rotate(deskew_angle, resample=Image.BICUBIC, expand=expand, fillcolor=(255,255,255))
    return (pil_rot, abs(angle)) if return_angle else (pil_rot, None)


def normalize_for_recognition(
    crop_img: Image.Image,
    *,
    target_h: int = 64,
    max_w: int = 1024,
    channel_first: bool = True,
) -> np.ndarray:
    """
    Resize and normalize a crop for recognition models.

    - Preserves aspect ratio, resizes height -> target_h.
    - Clips width to max_w.
    - Converts to float32 in [0, 1].
    - Returns shape (C,H,W) if channel_first True, else (H,W,C).

    This is intentionally minimal and leaves aggressive augmentation to augment_for_recognition.
    """
    if target_h <= 0:
        raise ValueError("target_h must be > 0")

    w, h = crop_img.size
    scale = float(target_h) / float(h)
    new_w = int(max(1, round(w * scale)))
    new_w = min(new_w, max_w)

    # resize using high-quality Lanczos
    resized = crop_img.resize((new_w, target_h), Image.LANCZOS).convert("RGB")
    arr = np.asarray(resized).astype(np.float32) / 255.0  # H,W,3

    if channel_first:
        arr = np.transpose(arr, (2, 0, 1))  # C,H,W

    return arr


def augment_for_recognition(
    crop_img: Image.Image,
    *,
    rotate_max_deg: float = 5.0,
    add_noise_std: float = 0.01,
    blur_radius: float = 0.0,
    contrast_factor: Optional[float] = None,
    seed: Optional[int] = None,
) -> Image.Image:
    """
    Apply small augmentations suitable for recognition training.

    rotate_max_deg: maximum absolute rotation in degrees (random uniform [-rotate_max_deg, rotate_max_deg]).
    add_noise_std: gaussian noise std (applied to normalized array).
    blur_radius: PIL GaussianBlur radius.
    contrast_factor: if provided, multiply contrast by a factor randomly sampled around this value:
                     the function picks uniform[1-contrast_factor, 1+contrast_factor].
    seed: optional random seed for deterministic augmentation.
    """
    if seed is not None:
        rnd = random.Random(seed)
        np_rand_state = np.random.get_state()
        np.random.seed(seed)
    else:
        rnd = random.Random()

    img = crop_img.convert("RGB")

    # rotation
    if rotate_max_deg and rotate_max_deg > 0:
        angle = rnd.uniform(-rotate_max_deg, rotate_max_deg)
        img = img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=(255,255,255))

    # blur
    if blur_radius and blur_radius > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # contrast
    if contrast_factor is not None and contrast_factor > 0:
        from PIL import ImageEnhance
        factor = rnd.uniform(max(0.1, 1.0 - contrast_factor), 1.0 + contrast_factor)
        img = ImageEnhance.Contrast(img).enhance(factor)

    # noise
    if add_noise_std and add_noise_std > 0:
        arr = np.asarray(img).astype(np.float32) / 255.0
        noise = np.random.normal(loc=0.0, scale=add_noise_std, size=arr.shape).astype(np.float32)
        arr = np.clip(arr + noise, 0.0, 1.0)
        img = Image.fromarray((arr * 255.0).astype(np.uint8))

    if seed is not None:
        np.random.set_state(np_rand_state)

    return img
