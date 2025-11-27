#!/usr/bin/env python3
"""
Preprocessing pipeline for scanned PDF pages:
- grayscale, denoise, deskew, CLAHE, adaptive threshold / binarize

Usage:
  python detector/preprocess.py --input detector/data/images/somepaper --output detector/data/images_preprocessed --deskew --clahe
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple
import argparse
from tqdm import tqdm
from PIL import Image

def load_image_cv(path: Path):
    im = Image.open(path).convert("RGB")
    arr = np.array(im)
    return arr

def save_image_cv(arr: np.ndarray, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(out_path)

def deskew_image(gray: np.ndarray) -> np.ndarray:
    # Use moments approach on edges; fallback to minAreaRect on text contours
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    edges = cv2.Canny(blur, 50, 150)
    coords = np.column_stack(np.where(edges > 0))
    if coords.shape[0] < 10:
        return gray
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    rotated = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def preprocess_image(img_bgr: np.ndarray,
                     denoise: bool = True,
                     deskew: bool = True,
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
