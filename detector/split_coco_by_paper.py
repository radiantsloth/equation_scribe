#!/usr/bin/env python3
"""
Split a COCO annotations file into train/val by paper.

Assumptions:
- Each image file_name encodes the paper id so we can group images by paper.
  The script supports two conventions:
    - "paperid_page_0001.png"
    - "paperid/page_0001.png" or "paperid\\page_0001.png"

Usage:
    python detector/split_coco_by_paper.py --coco detector/data/annotations/instances_all.json \
        --out-dir detector/data/annotations --val-frac 0.2 --seed 0
"""
import json
import argparse
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

PAPER_REGEX = re.compile(r"^(.+?)_page_\d{1,4}\.")  # matches paperid_page_0001.png


def infer_paper_from_filename(fname: str) -> str:
    """Try multiple heuristic patterns to obtain a paper_id from an image filename."""
    base = Path(fname).name
    m = PAPER_REGEX.match(base)
    if m:
        return m.group(1)
    # fallback: try folder name if path includes directories
    p = Path(fname)
    if len(p.parts) >= 2:
        return p.parts[-2]
    # fallback: use stem before first underscore
    if "_" in p.stem:
        return p.stem.split("_")[0]
    return p.stem


def split_coco_by_paper(coco_in: Path, out_dir: Path, val_frac: float = 0.2, seed: int = 0):
    coco = json.load(open(coco_in, "r", encoding="utf-8"))
    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])

    # build mapping image_id -> image dict
    images_by_id: Dict[int, Dict] = {img["id"]: img for img in images}

    # group image ids by paper
    paper_to_image_ids: Dict[str, List[int]] = {}
    for img in images:
        paper_id = infer_paper_from_filename(img["file_name"])
        paper_to_image_ids.setdefault(paper_id, []).append(img["id"])

    # shuffle and split papers
    papers = list(paper_to_image_ids.keys())
    random.Random(seed).shuffle(papers)
    nval = max(1, int(len(papers) * val_frac))
    val_papers = set(papers[:nval])
    train_papers = set(papers[nval:])

    def build_subset(paperset):
        out_images = []
        out_annotations = []
        image_ids = set()
        for paper in paperset:
            ids = paper_to_image_ids.get(paper, [])
            for iid in ids:
                out_images.append(images_by_id[iid])
                image_ids.add(iid)
        for ann in annotations:
            if ann["image_id"] in image_ids:
                out_annotations.append(ann)
        return out_images, out_annotations

    train_images, train_annotations = build_subset(train_papers)
    val_images, val_annotations = build_subset(val_papers)

    train_coco = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "images": train_images,
        "annotations": train_annotations,
        "categories": categories,
    }
    val_coco = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "images": val_images,
        "annotations": val_annotations,
        "categories": categories,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "instances_train.json"
    val_path = out_dir / "instances_val.json"
    json.dump(train_coco, open(train_path, "w", encoding="utf-8"), indent=2)
    json.dump(val_coco, open(val_path, "w", encoding="utf-8"), indent=2)
    print(f"Wrote: {train_path} ({len(train_images)} images, {len(train_annotations)} anns)")
    print(f"Wrote: {val_path} ({len(val_images)} images, {len(val_annotations)} anns)")
    return train_path, val_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--coco", required=True, help="Input COCO JSON (instances_all.json)")
    p.add_argument("--out-dir", required=True, help="Output directory for instances_train.json / instances_val.json")
    p.add_argument("--val-frac", default=0.2, type=float)
    p.add_argument("--seed", default=0, type=int)
    args = p.parse_args()
    split_coco_by_paper(Path(args.coco), Path(args.out_dir), args.val_frac, args.seed)


if __name__ == "__main__":
    main()
