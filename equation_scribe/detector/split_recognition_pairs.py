#!/usr/bin/env python3
"""
Split a recognition JSONL (image->latex) into train / val sets.

Supports multiple splitting strategies:

1. --group-by paper (default when --coco is provided)
   Uses a COCO file (e.g. instances_all.json) to map image_id -> paper_id, groups
   all crops from the same paper into one split to avoid leaking similar pages.

2. --group-by page
   Groups by page/image_id (all crops from same image go to same split).

3. --group-by random
   Randomly split entries (not recommended unless grouping is unnecessary).

4. --use-coco-split
   If you already have COCO train/val files (instances_tiles_train.json and
   instances_tiles_val.json), the script will place recognition crops whose
   image_id is in the COCO-train into train, those in COCO-val into val.

Crop filenames produced by `make_recognition_pairs.py` default to the form:
    crop_img{img_id:06d}_ann{ann_id:06d}.png
The script attempts to extract `img_id` by regex `img(\d+)`. If your naming
differs, the script will fall back to random splitting for unmatched items.

Usage examples:

# Simple random split
python -m equation_scribe.detector.split_recognition_pairs \
  --rec detector/data/recognition_pairs/all.jsonl \
  --out-train detector/data/recognition_pairs/train.jsonl \
  --out-val detector/data/recognition_pairs/val.jsonl \
  --val-frac 0.2 --seed 123 --group-by random

# Split by paper using instances_all.json
python -m equation_scribe.detector.split_recognition_pairs \
  --rec detector/data/recognition_pairs/all.jsonl \
  --coco detector/data/annotations/instances_all.json \
  --out-train detector/data/recognition_pairs/train.jsonl \
  --out-val detector/data/recognition_pairs/val.jsonl \
  --val-frac 0.2 --seed 123 --group-by paper

# Use an existing COCO train/val split (maps recognition crops to train/val by img_id)
python -m equation_scribe.detector.split_recognition_pairs \
  --rec detector/data/recognition_pairs/all.jsonl \
  --coco-train detector/data/annotations/instances_tiles_train.json \
  --coco-val detector/data/annotations/instances_tiles_val.json \
  --out-train detector/data/recognition_pairs/train.jsonl \
  --out-val detector/data/recognition_pairs/val.jsonl \
  --use-coco-split

"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

IMGID_RE = re.compile(r'img(\d+)', re.IGNORECASE)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rec", required=True, help="Recognition JSONL file (each line {'image':..., 'text':...})")
    p.add_argument("--coco", help="COCO file with images metadata (instances_all.json); used for paper grouping.")
    p.add_argument("--coco-train", help="COCO train file (instances_tiles_train.json) for use-coco-split")
    p.add_argument("--coco-val", help="COCO val file (instances_tiles_val.json) for use-coco-split")
    p.add_argument("--out-train", required=True, help="Output JSONL for train")
    p.add_argument("--out-val", required=True, help="Output JSONL for val")
    p.add_argument("--val-frac", type=float, default=0.2, help="Fraction for validation (if not using coco split)")
    p.add_argument("--seed", type=int, default=0, help="Random seed for reproducible splits")
    p.add_argument("--group-by", choices=["paper", "page", "random"], default="paper",
                   help="Grouping/stratification choice. 'paper' groups by paper_id (requires --coco).")
    p.add_argument("--use-coco-split", action="store_true",
                   help="If provided with --coco-train and --coco-val, use those to assign crops.")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def read_jsonl(path: Path) -> List[Dict]:
    recs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            recs.append(json.loads(line))
    return recs


def write_jsonl(path: Path, records: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_coco_images(coco_path: Path) -> Dict[int, Dict]:
    with coco_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)
    images = {int(im["id"]): im for im in coco.get("images", [])}
    return images


def build_imgid_from_croppath(crop_path: str) -> Tuple[int, bool]:
    """
    Try to extract img_id from crop filename. Returns (img_id, True) on success,
    (None, False) on failure.
    """
    m = IMGID_RE.search(Path(crop_path).name)
    if m:
        return int(m.group(1)), True
    return None, False


def split_by_coco_train_val(recs: List[Dict], coco_train_path: Path, coco_val_path: Path, verbose=False):
    train_ids = set(load_coco_images(coco_train_path).keys())
    val_ids = set(load_coco_images(coco_val_path).keys())

    train_out, val_out, unknown = [], [], []
    for r in recs:
        imgid, ok = build_imgid_from_croppath(r["image"])
        if not ok:
            unknown.append(r)
            continue
        if imgid in train_ids:
            train_out.append(r)
        elif imgid in val_ids:
            val_out.append(r)
        else:
            unknown.append(r)

    if unknown and verbose:
        print(f"[WARN] {len(unknown)} recognition records could not be mapped to coco-train/val sets. Placing in train by default.")
        train_out.extend(unknown)
    return train_out, val_out


def split_grouped_by_paper(recs: List[Dict], coco_images: Dict[int, Dict], val_frac: float, seed: int, verbose=False):
    # Build mapping: img_id -> paper_id
    imgid_to_paper = {}
    for img_id, im in coco_images.items():
        # Prefer explicit 'paper_id' in coco image metadata, else fallback to file_name
        paper = im.get("paper_id") or im.get("file_name")
        imgid_to_paper[int(img_id)] = paper

    # Group recognition records by paper_id
    groups = defaultdict(list)
    unknown = []
    for r in recs:
        imgid, ok = build_imgid_from_croppath(r["image"])
        if not ok:
            unknown.append(r)
            continue
        paper = imgid_to_paper.get(imgid)
        if paper is None:
            unknown.append(r)
            continue
        groups[paper].append(r)

    papers = list(groups.keys())
    random.Random(seed).shuffle(papers)
    n_val = max(1, int(round(val_frac * len(papers))))
    val_papers = set(papers[:n_val])

    train_out, val_out = [], []
    for paper, items in groups.items():
        if paper in val_papers:
            val_out.extend(items)
        else:
            train_out.extend(items)

    # Put unknowns into train (could also choose random assignment)
    if unknown and verbose:
        print(f"[WARN] {len(unknown)} recognition records unmapped to image/paper; adding to train.")
        train_out.extend(unknown)

    if verbose:
        print(f"split: {len(papers)} papers => {len(val_papers)} val papers; counts train={len(train_out)} val={len(val_out)}")
    return train_out, val_out


def split_grouped_by_page(recs: List[Dict], val_frac: float, seed: int, verbose=False):
    # Group by img_id
    groups = defaultdict(list)
    unknown = []
    for r in recs:
        imgid, ok = build_imgid_from_croppath(r["image"])
        if not ok:
            unknown.append(r)
            continue
        groups[imgid].append(r)
    pages = list(groups.keys())
    random.Random(seed).shuffle(pages)
    n_val = max(1, int(round(val_frac * len(pages))))
    val_pages = set(pages[:n_val])

    train_out, val_out = [], []
    for page, items in groups.items():
        if page in val_pages:
            val_out.extend(items)
        else:
            train_out.extend(items)
    if unknown and verbose:
        print(f"[WARN] {len(unknown)} recognition records unmapped to page; adding to train.")
        train_out.extend(unknown)
    if verbose:
        print(f"split: {len(pages)} pages => {len(val_pages)} val pages; counts train={len(train_out)} val={len(val_out)}")
    return train_out, val_out


def split_random(recs: List[Dict], val_frac: float, seed: int):
    rng = random.Random(seed)
    rng.shuffle(recs)
    n_val = max(1, int(round(val_frac * len(recs))))
    val = recs[:n_val]
    train = recs[n_val:]
    return train, val


def main():
    args = parse_args()
    recs = read_jsonl(Path(args.rec))
    if args.verbose:
        print(f"Read {len(recs)} recognition records from {args.rec}")

    # If user asked to use an existing coco train/val split
    if args.use_coco_split:
        if not (args.coco_train and args.coco_val):
            raise ValueError("--use-coco-split requires --coco-train and --coco-val")
        train_out, val_out = split_by_coco_train_val(recs, Path(args.coco_train), Path(args.coco_val), verbose=args.verbose)
        write_jsonl(Path(args.out_train), train_out)
        write_jsonl(Path(args.out_val), val_out)
        if args.verbose:
            print(f"Wrote train={len(train_out)} val={len(val_out)}")
        return

    # If grouping by paper, require COCO images
    if args.group_by == "paper":
        if not args.coco:
            raise ValueError("--group-by paper requires --coco (COCO file with 'paper_id' in images)")
        coco_images = load_coco_images(Path(args.coco))
        train_out, val_out = split_grouped_by_paper(recs, coco_images, args.val_frac, args.seed, verbose=args.verbose)
    elif args.group_by == "page":
        train_out, val_out = split_grouped_by_page(recs, args.val_frac, args.seed, verbose=args.verbose)
    else:  # random
        train_out, val_out = split_random(recs, args.val_frac, args.seed)

    write_jsonl(Path(args.out_train), train_out)
    write_jsonl(Path(args.out_val), val_out)
    if args.verbose:
        print(f"Wrote train={len(train_out)} val={len(val_out)}")


if __name__ == "__main__":
    main()
