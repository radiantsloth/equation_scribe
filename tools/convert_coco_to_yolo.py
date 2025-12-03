#!/usr/bin/env python3
"""
Convert a COCO-style tiled annotations JSON into YOLO-format labels.

Writes labels under: <dataset_root>/labels/<images-subdir>/<image_basename>.txt
For example:
  detector/data/labels/tiles_train/paper001_page_0005_tile_0000.txt

Each label file contains lines:
  <class_idx> <x_center_norm> <y_center_norm> <width_norm> <height_norm>

Usage:
  python tools/convert_coco_to_yolo.py \
    --coco detector/data/annotations/instances_tiles_train.json \
    --dataset-root detector/data \
    --out-labels detector/data/labels
"""
import argparse
import json
from pathlib import Path
from collections import defaultdict
import math

def convert(coco_path: Path, dataset_root: Path, out_labels_root: Path):
    coco = json.load(open(coco_path, "r", encoding="utf-8"))
    images = {img["id"]: img for img in coco.get("images", [])}
    anns = coco.get("annotations", [])
    cats = coco.get("categories", [])
    # Map category id to yolo class index (0-based). We will map sorted unique category ids to indices.
    cat_ids = sorted({c["id"] for c in cats})
    catid_to_idx = {cid: i for i, cid in enumerate(cat_ids)}

    # Group annotations by image_id
    ann_by_img = defaultdict(list)
    for a in anns:
        ann_by_img[a["image_id"]].append(a)

    out_labels_root.mkdir(parents=True, exist_ok=True)

    written = 0
    missing_image_count = 0
    for img_id, img in images.items():
        file_name = img["file_name"]  # e.g., images/tiles_train/xxx.png
        img_path = (dataset_root / file_name).resolve()
        if not img_path.exists():
            # sometimes file_name may be absolute already
            if Path(file_name).exists():
                img_path = Path(file_name).resolve()
            else:
                print(f"[WARN] Image file missing for COCO image entry: {file_name}")
                missing_image_count += 1
                continue
        iw, ih = img.get("width"), img.get("height")
        if not iw or not ih:
            # fallback to reading from file? For now skip
            print(f"[WARN] No width/height for image {file_name}; skipping")
            continue

        # label output path: out_labels_root / <images-subdir> / basename.txt
        img_rel = Path(file_name)
        # e.g., images/tiles_train/paper... => we want labels/tiles_train/<basename>.txt
        # so use suffix of img_rel: drop leading 'images/' if present
        parts = img_rel.parts
        if parts[0] == "images":
            subpath = Path(*parts[1:])  # tiles_train/...
        else:
            subpath = img_rel
        label_dir = out_labels_root / subpath.parent
        label_dir.mkdir(parents=True, exist_ok=True)
        label_path = label_dir / (img_rel.stem + ".txt")

        lines = []
        for a in ann_by_img.get(img_id, []):
            cid = a.get("category_id", 0)
            cls_idx = catid_to_idx.get(cid, 0)
            bx, by, bw, bh = a["bbox"]  # COCO: x,y,width,height (pixels)
            # convert to xy_center normalized
            xc = bx + bw / 2.0
            yc = by + bh / 2.0
            xcn = xc / float(iw)
            ycn = yc / float(ih)
            bwn = bw / float(iw)
            bhn = bh / float(ih)
            # ensure values in (0,1)
            xcn = min(max(xcn, 0.0), 1.0)
            ycn = min(max(ycn, 0.0), 1.0)
            bwn = min(max(bwn, 0.0), 1.0)
            bhn = min(max(bhn, 0.0), 1.0)

            lines.append(f"{cls_idx} {xcn:.6f} {ycn:.6f} {bwn:.6f} {bhn:.6f}")

        # write label file (empty file OK if no annotations)
        with open(label_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
        written += 1

    print(f"Wrote {written} label files under {out_labels_root.resolve()}")
    if missing_image_count:
        print(f"{missing_image_count} images were missing on disk and skipped.")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--coco", required=True)
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--out-labels", required=True)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    convert(Path(args.coco), Path(args.dataset_root), Path(args.out_labels))
