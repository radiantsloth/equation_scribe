#!/usr/bin/env python3
"""
detector/make_pairs.py

Given a COCO annotations file and the images referenced therein, crop each bbox
and write recognition pairs (image -> latex) as PNG files and a `pairs.jsonl`.

Usage:
  python detector/make_pairs.py --coco detector/data/annotations/instances_all.json \
    --out-images detector/data/recognition/images --out-jsonl detector/data/recognition/pairs.jsonl
"""
from pathlib import Path
import argparse
import json
from PIL import Image
import os
from tqdm import tqdm

def crop_and_save(img_path: Path, bbox, out_dir: Path, prefix: str):
    x, y, w, h = bbox
    x0, y0, x1, y1 = int(x), int(y), int(x + w), int(y + h)
    with Image.open(img_path) as im:
        W, H = im.size
        # clip
        x0 = max(0, min(x0, W-1))
        y0 = max(0, min(y0, H-1))
        x1 = max(1, min(x1, W))
        y1 = max(1, min(y1, H))
        crop = im.crop((x0, y0, x1, y1))
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = f\"{prefix}_{x0}_{y0}_{x1}_{y1}.png\"
        out_path = out_dir / fname
        crop.save(out_path)
        return out_path

def coco_to_pairs(coco_json: Path, out_images: Path, out_jsonl: Path, page_images_root: Path = None, pair_prefix=\"pair\"):
    coco = json.load(open(coco_json, \"r\", encoding=\"utf-8\"))
    images = {img[\"id\"]: img for img in coco.get(\"images\", [])}
    anns = coco.get(\"annotations\", [])
    # Optional: if your images paths are relative, join with a root
    pairs = []
    count = 0
    for ann in tqdm(anns, desc=\"cropping\"):
        img = images[ann[\"image_id\"]]
        img_path = Path(img[\"file_name\"])
        if page_images_root and not img_path.exists():
            candidate = Path(page_images_root) / img_path.name
            if candidate.exists():
                img_path = candidate
        if not img_path.exists():
            # skip missing image
            continue
        # latex text not stored in COCO; if you built COCO from synthetic generator we don't have latex here.
        # Optionally: support a companion JSON that maps image+ann->latex. Here we expect the COCO 'caption' or
        # the image-side json file to hold mapping. We'll look for an optional .meta.json per image.
        meta_json = img_path.with_suffix('.meta.json')
        latex = None
        if meta_json.exists():
            md = json.load(open(meta_json, 'r', encoding='utf-8'))
            # find the latex whose bbox matches (simple bounding)
            for rec in md.get('eqs', []):
                # if bbox close to ann bbox, pick it
                bx0, by0, bx1, by1 = rec['bbox']
                ax0, ay0, aw, ah = ann['bbox']
                ax1, ay1 = ax0+aw, ay0+ah
                # check IoU-ish
                iox0 = max(bx0, ax0); ioy0 = max(by0, ay0)
                iox1 = min(bx1, ax1); ioy1 = min(by1, ay1)
                if iox1 > iox0 and ioy1 > ioy0:
                    latex = rec.get('latex')
                    break
        # If no latex available, we skip (recognizer needs pairs)
        if not latex:
            # skip if no gold latex
            continue
        out_path = crop_and_save(img_path, ann['bbox'], out_images, f\"{pair_prefix}_{ann['id']}\")
        pairs.append({\"image\": str(out_path), \"latex\": latex})
        count += 1
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_jsonl, 'w', encoding='utf-8') as fh:
        for p in pairs:
            fh.write(json.dumps(p, ensure_ascii=False) + '\\n')
    print(f\"Wrote {count} pairs to {out_jsonl}\")
    return out_jsonl

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--coco', required=True)
    p.add_argument('--out-images', required=True)
    p.add_argument('--out-jsonl', required=True)
    p.add_argument('--page-images-root', default=None, help='Optional root to resolve relative image paths')
    args = p.parse_args()
    coco_to_pairs(Path(args.coco), Path(args.out_images), Path(args.out_jsonl), page_images_root=Path(args.page_images_root) if args.page_images_root else None)

if __name__ == '__main__':
    main()
