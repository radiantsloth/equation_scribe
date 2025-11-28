#!/usr/bin/env python3
"""
detector/make_pairs.py

Given a COCO annotations file and the images referenced therein, crop each bbox
and write recognition pairs (image -> latex) as PNG files and a `pairs.jsonl`.

This script expects that for synthetic data, each image has a companion
`<image>.meta.json` file containing `eqs` with `latex` and `bbox`. If that meta
file is missing the annotation will be skipped (recognizer needs gold LaTeX).
"""
from pathlib import Path
import argparse
import json
from PIL import Image
from tqdm import tqdm

def crop_and_save(img_path: Path, bbox, out_dir: Path, prefix: str):
    x, y, w, h = bbox
    x0, y0, x1, y1 = int(round(x)), int(round(y)), int(round(x + w)), int(round(y + h))
    with Image.open(img_path) as im:
        W, H = im.size
        # clip to image bounds
        x0 = max(0, min(x0, W-1))
        y0 = max(0, min(y0, H-1))
        x1 = max(1, min(x1, W))
        y1 = max(1, min(y1, H))
        if x1 <= x0 or y1 <= y0:
            return None
        crop = im.crop((x0, y0, x1, y1))
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{prefix}_{x0}_{y0}_{x1}_{y1}.png"
        out_path = out_dir / fname
        crop.save(out_path)
        return out_path

def find_latex_for_annotation(img_meta: dict, ann_bbox):
    """
    Given a page meta JSON (with eqs), find the latex for the annotation by bbox overlap.
    Returns the latex string or None.
    """
    ax, ay, aw, ah = ann_bbox
    ax1, ay1 = ax + aw, ay + ah
    best = None
    best_iou = 0.0
    for rec in img_meta.get("eqs", []):
        bx0, by0, bx1, by1 = rec.get("bbox", [0,0,0,0])
        bw, bh = bx1 - bx0, by1 - by0
        # compute intersection
        ix0 = max(ax, bx0); iy0 = max(ay, by0)
        ix1 = min(ax1, bx1); iy1 = min(ay1, by1)
        if ix1 <= ix0 or iy1 <= iy0:
            continue
        inter = (ix1 - ix0) * (iy1 - iy0)
        ann_area = aw * ah
        if ann_area <= 0:
            continue
        iou_like = inter / ann_area
        if iou_like > best_iou:
            best_iou = iou_like
            best = rec.get("latex")
    # require a minimum overlap
    if best_iou >= 0.25:
        return best
    return None

def coco_to_pairs(coco_json: Path, out_images: Path, out_jsonl: Path, page_images_root: Path = None, pair_prefix="pair"):
    coco = json.load(open(coco_json, "r", encoding="utf-8"))
    images = {img["id"]: img for img in coco.get("images", [])}
    anns = coco.get("annotations", [])
    pairs = []
    count = 0
    for ann in tqdm(anns, desc="cropping"):
        img = images.get(ann["image_id"])
        if img is None:
            continue
        img_path = Path(img["file_name"])
        # resolve relative path under page_images_root if necessary
        if page_images_root and not img_path.exists():
            candidate = Path(page_images_root) / img_path.name
            if candidate.exists():
                img_path = candidate
        if not img_path.exists():
            # try metadata path as fallback
            print(f"Warning: image file not found: {img['file_name']}; skipping")
            continue

        meta_path = img_path.with_suffix(".meta.json")
        img_meta = None
        if meta_path.exists():
            try:
                img_meta = json.load(open(meta_path, "r", encoding="utf-8"))
            except Exception:
                img_meta = None

        latex = None
        if img_meta:
            latex = find_latex_for_annotation(img_meta, ann['bbox'])

        if not latex:
            # No gold latex found; skip this annotation
            continue

        out_path = crop_and_save(img_path, ann['bbox'], out_images, f"{pair_prefix}_{ann['id']}")
        if out_path:
            pairs.append({"image": str(out_path), "latex": latex})
            count += 1

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as fh:
        for p in pairs:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Wrote {count} pairs to {out_jsonl}")
    return out_jsonl

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--coco', required=True)
    p.add_argument('--out-images', required=True)
    p.add_argument('--out-jsonl', required=True)
    p.add_argument('--page-images-root', default=None, help='Optional root to resolve relative image paths')
    args = p.parse_args()
    coco_to_pairs(Path(args.coco), Path(args.out_images), Path(args.out_jsonl),
                  page_images_root=Path(args.page_images_root) if args.page_images_root else None)

if __name__ == '__main__':
    main()
