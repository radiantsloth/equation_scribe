# tools/preprocess_recognition_images.py
"""
Preprocess recognition images to ViT input size and optionally augment.

Usage:
python tools/preprocess_recognition_images.py \
  --jsonl detector/data/recognition_pairs/im2latex_train.jsonl \
  --images-root data/im2latex/formula_images_processed/formula_images_processed \
  --out-dir detector/data/recognition_pairs/crops_vit224 \
  --size 224 \
  --augment \
  --seed 123
"""
import argparse
from pathlib import Path
from PIL import Image, ImageOps, ImageFilter
import json
import random
import io
import os
import numpy as np

def resize_and_pad(img: Image.Image, size: int, pad_color=(255,255,255)):
    w,h = img.size
    scale = min(size / w, size / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    img_resized = img.resize((new_w, new_h), Image.LANCZOS)
    # pad to size centered
    delta_w = size - new_w
    delta_h = size - new_h
    pad = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
    out = ImageOps.expand(img_resized, border=pad, fill=pad_color)
    return out

def random_augment(pil_img: Image.Image, rotate_max=12, deskew_prob=0.3):
    # simple augmentations: small rotation, gaussian blur, jpeg compression, noise
    img = pil_img
    # random rotation (centered)
    if random.random() < 0.7:
        ang = random.uniform(-rotate_max, rotate_max)
        img = img.rotate(ang, resample=Image.BICUBIC, expand=False, fillcolor=(255,255,255))

    # gaussian blur
    if random.random() < 0.2:
        r = random.uniform(0.5, 1.8)
        img = img.filter(ImageFilter.GaussianBlur(radius=r))

    # jpeg compression artifact: save to buffer with random quality and reload
    if random.random() < 0.3:
        buf = io.BytesIO()
        q = random.randint(40, 95)
        img.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")

    # add small gaussian noise using numpy
    if random.random() < 0.2:
        arr = np.asarray(img).astype(np.float32)
        sigma = random.uniform(2.0, 12.0)
        noise = np.random.normal(0, sigma, arr.shape).astype(np.float32)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    return img

def preprocess_jsonl(jsonl_path: Path, images_root: Path, out_dir: Path, size:int=224, do_augment=False, max_records=None, seed=42):
    random.seed(seed)
    images_root = images_root.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    in_count = 0
    out_records = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            in_count += 1
            if max_records and in_count > max_records:
                break
            img_path = Path(rec["image"])
            # if image path is relative, join with images_root
            if not img_path.is_absolute():
                cand = images_root / img_path
            else:
                cand = img_path
            if not cand.exists():
                # try basename lookup under images_root
                bas = img_path.name
                cand = next(images_root.rglob(bas), None)
                if cand is None or not cand.exists():
                    print(f"Skipping missing image: {rec['image']}")
                    continue
            try:
                img = Image.open(cand).convert("RGB")
            except Exception as e:
                print("Could not open:", cand, e)
                continue

            if do_augment:
                img = random_augment(img)

            p = resize_and_pad(img, size=size, pad_color=(255,255,255))

            # save: maintain folder structure? We'll write all into out_dir
            out_fname = f"{jsonl_path.stem}_{in_count:06d}_{cand.name}"
            out_path = out_dir / out_fname
            p.save(out_path, format="PNG")
            new_rec = dict(rec)  # shallow copy
            new_rec["image"] = str(out_path)
            out_records.append(new_rec)

    # write new jsonl
    out_jsonl = out_dir.parent / f"{jsonl_path.stem}_vit{size}.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as fo:
        for r in out_records:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Processed {len(out_records)} / {in_count} records -> images in {out_dir}, jsonl: {out_jsonl}")
    return out_jsonl

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", required=True)
    p.add_argument("--images-root", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--augment", action="store_true")
    p.add_argument("--max-records", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    preprocess_jsonl(Path(args.jsonl), Path(args.images_root), Path(args.out_dir), size=args.size,
                    do_augment=args.augment, max_records=args.max_records, seed=args.seed)
