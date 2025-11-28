#!/usr/bin/env python3
"""
Tile page images into overlapping crops and produce COCO annotations for tiles.

Usage:
  python detector/tiling.py --coco detector/data/annotations/instances_all.json \
     --images_root detector/data/images --out detector/data/tiles/instances_tiles.json \
     --tile-size 1024 --stride 512 --min-area-frac 0.25
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict
from PIL import Image
import random
import math
from tqdm import tqdm

def crop_and_collect_tiles(img_path: Path, anns_for_image: List[Dict],
                           tile_size: int = 1024, stride: int = 512,
                           min_area_frac: float = 0.25, keep_empty_prob: float = 0.05):
    im = Image.open(img_path)
    W, H = im.size
    tiles = []
    y = 0
    x_positions = list(range(0, max(1, W - tile_size + 1), stride))
    y_positions = list(range(0, max(1, H - tile_size + 1), stride))
    # ensure last tile touches edge
    if not x_positions or x_positions[-1] + tile_size < W:
        x_positions.append(max(0, W - tile_size))
    if not y_positions or y_positions[-1] + tile_size < H:
        y_positions.append(max(0, H - tile_size))

    tile_id = 0
    for y0 in y_positions:
        for x0 in x_positions:
            x1 = min(x0 + tile_size, W)
            y1 = min(y0 + tile_size, H)
            kept = []
            for ann in anns_for_image:
                bx, by, bw, bh = ann["bbox"]
                ax0, ay0, ax1, ay1 = bx, by, bx + bw, by + bh
                inter_x0 = max(x0, ax0)
                inter_y0 = max(y0, ay0)
                inter_x1 = min(x1, ax1)
                inter_y1 = min(y1, ay1)
                if inter_x1 <= inter_x0 or inter_y1 <= inter_y0:
                    continue
                inter_area = (inter_x1 - inter_x0) * (inter_y1 - inter_y0)
                ann_area = bw * bh
                if ann_area <= 0:
                    continue
                if (inter_area / ann_area) >= min_area_frac:
                    # compute adjusted bbox relative to tile top-left
                    nx0 = inter_x0 - x0
                    ny0 = inter_y0 - y0
                    nx1 = inter_x1 - x0
                    ny1 = inter_y1 - y0
                    nw = nx1 - nx0
                    nh = ny1 - ny0
                    kept.append({
                        "bbox": [nx0, ny0, nw, nh],
                        "category_id": ann["category_id"]
                    })
            # decide if we keep the tile
            if kept or (random.random() < keep_empty_prob):
                # save tile image path (no saving here, return info)
                tiles.append({
                    "tile_index": tile_id,
                    "tile_box": [x0, y0, x1, y1],
                    "annos": kept
                })
                tile_id += 1
    return tiles

def generate_tiles_from_coco(coco_in_path: Path, images_root: Path, out_images_dir: Path,
                             out_annotations_path: Path, tile_size=1024, stride=512,
                             min_area_frac=0.25, keep_empty_prob=0.05):
    coco = json.load(open(coco_in_path, "r", encoding="utf-8"))
    images = coco.get("images", [])
    anns = coco.get("annotations", [])
    categories = coco.get("categories", [])

    # build annotations by image_id
    ann_by_image = {}
    for a in anns:
        ann_by_image.setdefault(a["image_id"], []).append(a)

    out_images_dir = Path(out_images_dir)
    out_images_dir.mkdir(parents=True, exist_ok=True)
    tile_images_info = []
    tile_annotations = []
    next_image_id = 1
    next_ann_id = 1

    for img in tqdm(images, desc="Tiling images"):
        img_file = Path(img["file_name"])
        # try to resolve relative path under images_root
        candidate = Path(img["file_name"])
        if not candidate.exists():
            candidate = images_root / img_file.name
        if not candidate.exists():
            # try paper subfolder
            parts = img_file.parts
            if len(parts) > 1:
                candidate = images_root / img_file
        if not candidate.exists():
            print("Skipping missing image:", img["file_name"])
            continue
        anns_for_image = ann_by_image.get(img["id"], [])
        tiles = crop_and_collect_tiles(candidate, anns_for_image, tile_size=tile_size, stride=stride,
                                       min_area_frac=min_area_frac, keep_empty_prob=keep_empty_prob)
        # save tiles and build COCO
        for t in tiles:
            x0,y0,x1,y1 = t["tile_box"]
            tile_img = Image.open(candidate).crop((x0,y0,x1,y1))
            tile_name = f"{candidate.stem}_tile_{t['tile_index']:04d}.png"
            out_tile_path = out_images_dir / tile_name
            tile_img.save(out_tile_path)
            # image record
            w,h = tile_img.size
            tile_images_info.append({"id": next_image_id, "file_name": str(out_tile_path), "width": w, "height": h})
            # annotations for this tile
            for ann in t["annos"]:
                bbox = ann["bbox"]
                cat_id = ann["category_id"]
                tile_annotations.append({
                    "id": next_ann_id,
                    "image_id": next_image_id,
                    "category_id": cat_id,
                    "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                    "area": float(bbox[2] * bbox[3]),
                    "iscrowd": 0,
                    "segmentation": []
                })
                next_ann_id += 1
            next_image_id += 1

    coco_tiles = {
        "info": {"description": "Tiled dataset"},
        "licenses": [],
        "images": tile_images_info,
        "annotations": tile_annotations,
        "categories": categories
    }
    out_annotations_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(coco_tiles, open(out_annotations_path, "w", encoding="utf-8"), indent=2)
    print(f"Wrote tiled COCO: {out_annotations_path} images={len(tile_images_info)} anns={len(tile_annotations)}")
    return out_annotations_path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--coco", required=True, help="Input COCO JSON")
    p.add_argument("--images-root", required=True, help="Root folder containing images referenced in COCO")
    p.add_argument("--out-images", required=True, help="Folder to write tile images")
    p.add_argument("--out-annotations", required=True, help="Output COCO JSON for tiles")
    p.add_argument("--tile-size", type=int, default=1024)
    p.add_argument("--stride", type=int, default=512)
    p.add_argument("--min-area-frac", type=float, default=0.25)
    p.add_argument("--keep-empty-prob", type=float, default=0.05)
    args = p.parse_args()
    generate_tiles_from_coco(Path(args.coco), Path(args.images_root), Path(args.out_images),
                             Path(args.out_annotations), tile_size=args.tile_size, stride=args.stride,
                             min_area_frac=args.min_area_frac, keep_empty_prob=args.keep_empty_prob)

if __name__ == "__main__":
    main()
