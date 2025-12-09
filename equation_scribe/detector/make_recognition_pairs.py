#!/usr/bin/env python3
"""
Create JSONL recognition pairs from a COCO annotations file.

Given a COCO-style annotations file (instances_*.json) and the corresponding images
folder, this script will crop each annotation bbox from the original page image,
save the crop as a PNG under the provided output directory, and emit a JSONL file
where each line is {"image": "path/to/crop.png", "text": "<latex>"}.

Usage:
  python -m equation_scribe.detector.make_recognition_pairs \
      --coco detector/data/annotations/instances_all.json \
      --images detector/data/images/synth \
      --out-dir detector/data/recognition_pairs/crops \
      --jsonl detector/data/recognition_pairs/train.jsonl

Options:
  --pad-px    : optional padding (pixels) to add around bbox when cropping (default 4)
  --deskew    : try to deskew crop using preprocess.deskew_crop if available

Notes: requires Pillow. The script will create the out-dir if needed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

from PIL import Image

# Optional deskew import (best-effort)
try:
    from equation_scribe.detector.preprocess import deskew_crop
except Exception:
    try:
        from .preprocess import deskew_crop  # fallback relative import
    except Exception:
        deskew_crop = None


def crop_and_save(page_img_path: Path, bbox: Tuple[int, int, int, int], out_path: Path, pad_px: int = 4, deskew: bool = False):
    img = Image.open(page_img_path).convert('RGB')
    x0, y0, w, h = bbox
    x1 = x0 + w
    y1 = y0 + h
    # apply padding but clamp to image bounds
    x0p = max(0, int(round(x0 - pad_px)))
    y0p = max(0, int(round(y0 - pad_px)))
    x1p = min(img.width, int(round(x1 + pad_px)))
    y1p = min(img.height, int(round(y1 + pad_px)))
    if x1p <= x0p or y1p <= y0p:
        raise ValueError(f'Invalid crop box after padding: {(x0p, y0p, x1p, y1p)}')
    crop = img.crop((x0p, y0p, x1p, y1p))
    if deskew and deskew_crop is not None:
        try:
            crop, _ = deskew_crop(crop, return_angle=True, expand=True)
        except Exception:
            # If deskew fails, fall back to uncropped image
            pass
    out_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(out_path, format='PNG')
    return out_path


def build_pairs_from_coco(coco_json: Path, images_dir: Path, out_dir: Path, out_jsonl: Path, pad_px: int = 4, deskew: bool = False):
    with open(coco_json, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    images = {im['id']: im for im in coco.get('images', [])}
    anns = coco.get('annotations', [])

    out_dir = Path(out_dir)
    out_jsonl = Path(out_jsonl)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    cnt = 0
    with open(out_jsonl, 'w', encoding='utf-8') as outf:
        for ann in anns:
            img_id = ann['image_id']
            img_meta = images.get(img_id)
            if img_meta is None:
                print(f'WARNING: image_id {img_id} not found in images list; skipping annotation {ann.get("id")}')
                continue
            page_fname = img_meta['file_name']
            page_path = Path(images_dir) / page_fname
            if not page_path.exists():
                # Try with common subdirs (synth_pre, tiles_train etc) by searching inside images_dir.
                possible = list(Path(images_dir).rglob(page_fname))
                if possible:
                    page_path = possible[0]
                else:
                    print(f'WARNING: image file not found: {page_path}; skipping annotation {ann.get("id")}')
                    continue

            bbox = ann.get('bbox')  # COCO [x,y,w,h]
            if not bbox:
                print(f'WARNING: no bbox for ann {ann.get("id")}; skipping')
                continue

            latex = ann.get('latex') or ann.get('caption') or ''
            crop_fname = f"crop_img{img_id:06d}_ann{ann['id']:06d}.png"
            out_crop = out_dir / crop_fname
            try:
                crop_and_save(page_path, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])), out_crop, pad_px=pad_px, deskew=deskew)
            except Exception as e:
                print(f'WARNING: failed to crop {page_path} bbox={bbox} : {e}')
                continue
            rec = {"image": str(out_crop), "text": latex}
            outf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            cnt += 1

    print(f"Wrote {cnt} recognition pairs to {out_jsonl} with crops under {out_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--coco", required=True, help="COCO annotations JSON")
    p.add_argument("--images", required=True, help="Directory containing page images (as in COCO file)")
    p.add_argument("--out-dir", required=True, help="Directory to place cropped images")
    p.add_argument("--jsonl", required=True, help="Output JSONL path for recognition pairs")
    p.add_argument("--pad-px", type=int, default=4, help="Padding around bbox when cropping")
    p.add_argument("--deskew", action='store_true', help="Attempt to deskew crops (requires preprocess.deskew_crop)")
    args = p.parse_args()

    build_pairs_from_coco(Path(args.coco), Path(args.images), Path(args.out_dir), Path(args.jsonl), pad_px=args.pad_px, deskew=args.deskew)


if __name__ == '__main__':
    main()
