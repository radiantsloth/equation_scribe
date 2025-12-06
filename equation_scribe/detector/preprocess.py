#!/usr/bin/env python3
"""
preprocess.py — image preprocessing utilities for scanned pages.

Functions include:
  - deskew_image: estimate and correct page skew,
  - apply_clahe: adaptive histogram equalization (optional),
  - binarization helpers.

This module may use OpenCV (cv2) and requires numpy and Pillow.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple
import argparse
from tqdm import tqdm
from PIL import Image
import math

from equation_scribe.config import DESKEW_THRESHOLD_DEG

def load_image_cv(path: Path):
    im = Image.open(path).convert("RGB")
    arr = np.array(im)
    return arr

def save_image_cv(arr: np.ndarray, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(out_path)

def deskew_image(pil_img):
    """
    Estimate the skew angle of a page and rotate the image to correct it.

    The function uses a binary foreground mask (via Otsu thresholding) and
    cv2.minAreaRect on foreground coordinates to estimate the oriented bounding
    rectangle and its angle. The detected angle is normalized to degrees.

    Args:
        pil_img: PIL Image of the page (any mode).
    Returns:
        (rotated_img, angle_deg): rotated image (PIL.Image) and angle in degrees
                                 applied to deskew the image (positive means
                                 image was rotated by -angle_deg to deskew).
    Notes:
        - If the absolute angle is smaller than DESKEW_THRESHOLD_DEG, the
          function will return the original image and angle 0.0.
    """

    # Convert to grayscale numpy array (uint8)
    img = np.array(pil_img.convert("L"))
    # Binarize for edge detection
    _, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Invert because text is dark on light background; want foreground=white
    bw_inv = 255 - bw

    # Find coordinates of non-zero pixels
    coords = np.column_stack(np.where(bw_inv > 0))
    if coords.size == 0:
        # nothing to deskew
        return pil_img, 0.0

    rect = cv2.minAreaRect(coords)
    # rect = ((cx, cy), (w, h), angle)
    angle = rect[-1]
    # cv2.minAreaRect angle convention:
    # - if width < height: angle = angle
    # - if width >= height: angle = angle + 90
    # Normalize to [-90, 90)
    if rect[1][0] < rect[1][1]:
        angle_deg = angle
    else:
        angle_deg = angle + 90.0

    # If angle is tiny, skip deskew entirely (avoids small numerical jitter)
    if abs(angle_deg) < DESKEW_THRESHOLD_DEG:
        return pil_img, 0.0

    # Rotate the original PIL image by -angle_deg to deskew
    rotated = pil_img.rotate(-angle_deg, resample=Image.BICUBIC, expand=True, fillcolor=(255,255,255))
    return rotated, angle_deg

def preprocess_image(img_bgr: np.ndarray,
                     denoise: bool = True,
                     deskew: bool = False,
                     clahe: bool = True,
                     binarize: bool = True) -> np.ndarray:
    # ensure grayscale
    if img_bgr.ndim == 3:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_bgr.copy()

    if denoise:
        # fast NL means denoising (good for scanned pages)
        gray = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)

    if deskew:
        try:
            gray = deskew_image(gray)
        except Exception:
            pass

    if clahe:
        clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe_obj.apply(gray)

    if binarize:
        # adaptive threshold
        try:
            gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 25, 10)
        except Exception:
            # fallback global threshold
            _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return gray

def process_folder(in_dir: Path, out_dir: Path, **kwargs):
    in_dir = Path(in_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted([p for p in in_dir.glob("**/*") if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]])
    for p in tqdm(files, desc="Preprocessing images"):
        arr = load_image_cv(p)
        out = preprocess_image(arr, **kwargs)
        out_path = out_dir / p.name
        save_image_cv(out, out_path)
    print("Preprocessed", len(files), "images ->", out_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Folder with page images")
    parser.add_argument("--output", required=True, help="Folder to write preprocessed images")
    parser.add_argument("--deskew", action="store_true")
    parser.add_argument("--denoise", action="store_true")
    parser.add_argument("--clahe", action="store_true")
    parser.add_argument("--binarize", action="store_true")
    args = parser.parse_args()
    process_folder(Path(args.input), Path(args.output), denoise=args.denoise, deskew=args.deskew, clahe=args.clahe, binarize=args.binarize)

if __name__ == "__main__":
    main()
