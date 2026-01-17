#!/usr/bin/env python3
"""
synthetic_coco.py — generate synthetic pages with rendered LaTeX equations.

This script renders LaTeX expressions to images, composes them onto synthetic
"page" images, and emits COCO-format annotations suitable for object detection
training.

Usage (CLI):
    python -m equation_scribe.detector.synthetic_coco \
      --out-images detector/data/images/synth \
      --out-anns detector/data/annotations/instances_all.json \
      --n-pages 50 --eqs-per-page 6 --dpi 150 \
      --text-size 9 --eq-scale 2.0 --eq-padding 5
"""

from __future__ import annotations
import argparse
import json
import math
import os
import random
import tempfile
import csv
import re
from pathlib import Path
from typing import List, Tuple, Optional
import shutil
from PIL import Image
import numpy as np
import logging
from PIL import ImageDraw, ImageFont
from equation_scribe.config import (
    MAX_EQ_WIDTH_FRAC, MAX_EQ_HEIGHT_FRAC,
    NON_OVERLAP_IOU, MAX_PLACEMENT_ATTEMPTS, ROTATION_AUG_MAX_ANGLE,
    DEFAULT_DPI,PAGE_WIDTH_IN,PAGE_HEIGHT_IN
)

VOCAB = "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore et dolore magna aliqua ut enim ad minim veniam quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur excepteur sint occaecat cupidatat non proident sunt in culpa qui officia deserunt mollit anim id est laborum analysis data system model performance network neural learning algorithm distribution probability vector matrix function linear optimal".split()

_render_mathtext = None
try:
    from equation_scribe.detector.render_latex import render_mathtext as _render_mathtext
except Exception:
    try:
        from .render_latex import render_mathtext as _render_mathtext
    except Exception:
        _render_mathtext = None

try:
    from equation_scribe.detector.render_latex import _latex_render, HAVE_PDF2IMAGE
except Exception:
    _latex_render = None
    HAVE_PDF2IMAGE = False

if _render_mathtext is None:
    import matplotlib.pyplot as plt
    import matplotlib

    def _matplotlib_render(expr: str, out_path: str, dpi: int = 150, fontsize: int = 20):
        tex = expr
        if not (tex.startswith("$") and tex.endswith("$")):
            tex = f"${tex}$"
        fig = plt.figure(figsize=(0.01, 0.01))
        fig.text(0.0, 0.0, tex, fontsize=fontsize)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)

    def render_mathtext(expr: str, out_path: str, dpi: int = 150, prefer_latex: bool = False):
        return _matplotlib_render(expr, out_path, dpi=dpi, fontsize=10)

    _render_mathtext = render_mathtext

def clean_latex_formula(tex: str) -> str:
    """
    Sanitizes raw LaTeX using robust logic from validate_linear.py.
    """
    if not tex: return ""
    # 1. Strip Comments (Robust Backslash Counting)
    try:
        idx = 0
        while True:
            idx = tex.index("%", idx)
            backslashes = 0
            i = idx - 1
            while i >= 0 and tex[i] == "\\":
                backslashes += 1
                i -= 1
            if backslashes % 2 == 0:
                tex = tex[:idx]
                break
            else:
                idx += 1
    except ValueError:
        pass

    # 2. Strip Labels (Recursive Brace Counting)
    while True:
        match = re.search(r"\\label\s*\{", tex)
        if not match:
            break
        start_idx = match.start()
        open_brace_idx = match.end() - 1
        depth = 1
        end_idx = -1
        for i in range(open_brace_idx + 1, len(tex)):
            if tex[i] == '{':
                depth += 1
            elif tex[i] == '}':
                depth -= 1
            if depth == 0:
                end_idx = i
                break
        if end_idx != -1:
            tex = tex[:start_idx] + " " + tex[end_idx+1:]
        else:
            break
    return tex.strip()

def draw_fake_ieee_text(page_img: Image.Image, dpi: int = 150, margin_px: int = 50, col_gap_px: int = 30, text_size_pt: int = 10):
    """
    Draws random text in a two-column layout.
    """
    draw = ImageDraw.Draw(page_img)
    w, h = page_img.size
    col_width = (w - (2 * margin_px) - col_gap_px) // 2
    cols = [
        (margin_px, margin_px, margin_px + col_width, h - margin_px), 
        (margin_px + col_width + col_gap_px, margin_px, w - margin_px, h - margin_px)
    ]
    
    # Use user-provided text size
    body_pt = text_size_pt
    header_pt = text_size_pt + 2
    
    body_px = int(body_pt * (dpi / 72.0))
    header_px = int(header_pt * (dpi / 72.0))
    
    try:
        font = ImageFont.truetype("arial.ttf", body_px) 
        header_font = ImageFont.truetype("arial.ttf", header_px)
    except IOError:
        font = ImageFont.load_default()
        header_font = font

    for (x0, y0, x1, y1) in cols:
        cursor_y = y0
        while cursor_y < y1:
            is_header = random.random() < 0.1
            current_font = header_font if is_header else font
            num_words = random.randint(3, 10) if is_header else random.randint(20, 100)
            text = " ".join(random.choices(VOCAB, k=num_words)).capitalize()
            words = text.split()
            line = ""
            for word in words:
                test_line = line + word + " "
                bbox = draw.textbbox((0,0), test_line, font=current_font)
                text_w = bbox[2] - bbox[0]
                if text_w < (col_width - 10):
                    line = test_line
                else:
                    draw.text((x0, cursor_y), line, fill="black", font=current_font)
                    line = word + " "
                    cursor_y += (header_px + 2) if is_header else (body_px + 2)
                    if cursor_y > y1: break
            if cursor_y < y1:
                draw.text((x0, cursor_y), line, fill="black", font=current_font)
                cursor_y += (header_px + 4) if is_header else (body_px + 4)
            cursor_y += random.randint(5, 15)

def make_blank_page(width_px: int, height_px: int, color=(255,255,255)) -> Image.Image:
    return Image.new("RGB", (width_px, height_px), color=color)

def _iou(box_a, box_b):
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)
    inter_w = max(0, inter_x1 - inter_x0)
    inter_h = max(0, inter_y1 - inter_y0)
    inter_area = inter_w * inter_h
    area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
    union = area_a + area_b - inter_area
    return 0.0 if union <= 0 else inter_area / union

def _tight_bbox_from_rgba(img: Image.Image):
    """
    Return tight bbox (x0,y0,x1,y1) of content (Alpha or Intensity).
    """
    if img.mode == "RGBA":
        alpha = img.split()[3]
        bbox = alpha.getbbox()
        if bbox and bbox != (0, 0, img.width, img.height):
            return bbox
    gray = img.convert("L")
    mask = gray.point(lambda p: 255 if p < 250 else 0)
    bbox = mask.getbbox()
    if bbox: return bbox
    return (0, 0, img.width, img.height)

def place_and_annotate_on_page(page_img, eq_images, page_annotations, rotate_aug=False, rotate_max=15.0, require_non_overlap=True, margin_frac=0.05, max_attempts_per_box=1000, padding=5):
    PAGE_W, PAGE_H = page_img.size
    margin = int(round(margin_frac * PAGE_W))
    placed_boxes = []

    for latex, eq_img in eq_images:
        if eq_img.mode != "RGBA":
            eq_img = eq_img.convert("RGBA")

        angle = random.uniform(-rotate_max, rotate_max) if rotate_aug else 0.0
        if abs(angle) > 1e-6:
            rotated = eq_img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=(255,255,255,0))
        else:
            rotated = eq_img

        # 1. Get tight bbox of content
        tx0, ty0, tx1, ty1 = _tight_bbox_from_rgba(rotated)

        # 2. Add padding (clamping to image boundaries)
        rw, rh = rotated.size
        tx0 = max(0, tx0 - padding)
        ty0 = max(0, ty0 - padding)
        tx1 = min(rw, tx1 + padding)
        ty1 = min(rh, ty1 + padding)

        # 3. Crop with padding
        cropped = rotated.crop((tx0, ty0, tx1, ty1))
        w, h = cropped.size

        # Resize if too large
        max_w = int(PAGE_W * MAX_EQ_WIDTH_FRAC)
        max_h = int(PAGE_H * MAX_EQ_HEIGHT_FRAC)
        if w > max_w or h > max_h:
            scale = min(max_w / w, max_h / h)
            new_w = max(10, int(round(w * scale)))
            new_h = max(10, int(round(h * scale)))
            cropped = cropped.resize((new_w, new_h), Image.LANCZOS)
            w, h = cropped.size

        placed = False
        attempts = 0
        while not placed and attempts < max_attempts_per_box:
            attempts += 1
            x = random.randint(margin, max(margin, PAGE_W - margin - w))
            y = random.randint(margin, max(margin, PAGE_H - margin - h))
            cand = (x, y, x + w, y + h)
            
            if require_non_overlap:
                overlap = False
                for prev in placed_boxes:
                    if _iou(cand, prev) > NON_OVERLAP_IOU:
                        overlap = True
                        break
                if overlap: continue

            # Paste with transparency
            if cropped.mode == "RGBA":
                bg = Image.new("RGB", (w, h), (255, 255, 255))
                bg.paste(cropped, mask=cropped.split()[3])
                page_img.paste(bg, (x, y))
            else:
                page_img.paste(cropped.convert("RGB"), (x, y))

            page_annotations.append({"latex": latex, "bbox": [float(x), float(y), float(x + w), float(y + h)], "angle": float(angle)})
            placed_boxes.append(cand)
            placed = True

        if not placed:
            logger = logging.getLogger(__name__)
            logger.warning("Could not place equation (skipping): %s", latex[:20])

def generate_synthetic_coco(out_images: Path, out_anns: Path, n_pages: int = 50, n_papers: int = 5, eqs_per_page: int = 4, dpi: int = DEFAULT_DPI, seed: int = 0, formulas_file: Optional[Path] = None, text_size: int = 10, eq_scale: float = 2.0, eq_padding: int = 5, debug_mode: bool = False):
    random.seed(seed)
    PAGE_W = int(PAGE_WIDTH_IN * dpi)
    PAGE_H = int(PAGE_HEIGHT_IN * dpi)

    out_images = Path(out_images)
    out_anns = Path(out_anns)
    out_images.mkdir(parents=True, exist_ok=True)
    out_anns.parent.mkdir(parents=True, exist_ok=True)

    coco = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "equation"}]}
    img_id = 1
    ann_id = 1

    # Load Formulas
    all_formulas = []
    if formulas_file and Path(formulas_file).exists():
        try:
            with open(formulas_file, "r", encoding="utf-8") as f: raw_lines = f.readlines()
        except UnicodeDecodeError:
            with open(formulas_file, "r", encoding="latin-1") as f: raw_lines = f.readlines()
            
        if Path(formulas_file).suffix.lower() == ".csv":
            import csv
            reader = csv.reader(raw_lines)
            for row in reader:
                if row:
                    clean = clean_latex_formula(row[0])
                    if len(clean) > 5: all_formulas.append(clean)
        else:
            for line in raw_lines:
                clean = clean_latex_formula(line.strip())
                if len(clean) > 5: all_formulas.append(clean)
        print(f"Loaded {len(all_formulas)} clean formulas.")
    else:
        print("Warning: No formulas file provided, using samples.")
        all_formulas = [r"E=mc^2", r"\nabla \cdot E = \rho"]

    pages_per_paper = [n_pages // n_papers] * n_papers
    for i in range(n_pages % n_papers): pages_per_paper[i] += 1

    # Render at higher DPI to make equations larger and sharper relative to text
    render_dpi = int(dpi * eq_scale)
    
    # Retry configuration
    # In debug mode, we fail fast (1 attempt). In bulk mode, we retry (8 attempts).
    MAX_RETRIES = 1 if debug_mode else 8

    page_global_idx = 0
    for paper_idx in range(n_papers):
        for _ in range(pages_per_paper[paper_idx]):
            fname = f"paper{paper_idx:03d}_page_{page_global_idx:04d}.png"
            fpath = out_images / fname

            page_img = make_blank_page(PAGE_W, PAGE_H)
            
            draw_fake_ieee_text(page_img, dpi=dpi, text_size_pt=text_size) 
            
            eq_images = []
            
            # --- RENDER LOOP WITH RETRIES ---
            for _ in range(eqs_per_page):
                current_eq_img = None
                current_expr = ""
                
                # Retry loop: Try to find ONE valid equation
                for attempt in range(MAX_RETRIES):
                    expr = random.choice(all_formulas)
                    
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpf:
                        tmpname = tmpf.name
                    
                    try:
                        _render_mathtext(expr, tmpname, dpi=render_dpi, prefer_latex=True)
                        img = Image.open(tmpname)
                        img.load() # Force load
                        
                        # Validate it's not empty
                        if img.getbbox() is None:
                            raise ValueError("Rendered empty image")
                            
                        current_eq_img = img.convert("RGBA")
                        current_expr = expr
                        
                        # Cleanup success
                        try: os.unlink(tmpname)
                        except: pass
                        
                        # Found a good one, break the retry loop!
                        break 
                        
                    except Exception as e:
                        # Cleanup failure
                        try: os.unlink(tmpname)
                        except: pass
                        
                        if debug_mode:
                            print(f"[DEBUG-FAIL] Formula: {expr}")
                            print(f"[DEBUG-FAIL] Error: {e}")
                            # Stop retrying in debug mode
                            break
                        # Else: continue loop to pick a new equation
                
                # Check if we got an image after retries
                if current_eq_img is None:
                    # Final Fallback: Red Box
                    # This happens if all 8 retries fail, or if debug mode failed once.
                    if debug_mode:
                        print("[DEBUG] Generating RED BOX fallback.")
                    
                    current_eq_img = Image.new("RGBA", (int(dpi), int(dpi*0.5)), (255, 0, 0, 255))
                    current_expr = "RENDER_FAILURE"

                eq_images.append((current_expr, current_eq_img))

            # --- PLACEMENT ---
            page_records = []
            place_and_annotate_on_page(
                page_img=page_img,
                eq_images=eq_images,
                page_annotations=page_records,
                rotate_aug=args.rotate_aug,
                rotate_max=args.rotate_max,
                padding=eq_padding
            )

            for rec in page_records:
                x0, y0, x1, y1 = rec["bbox"]
                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 1,
                    "bbox": [int(x0), int(y0), int(x1-x0), int(y1-y0)],
                    "area": int((x1-x0)*(y1-y0)),
                    "iscrowd": 0,
                    "paper_id": f"paper{paper_idx:03d}",
                    "latex": rec["latex"],
                    "angle": rec.get("angle", 0.0),
                })
                ann_id += 1

            page_img.save(fpath, format="PNG", dpi=(dpi, dpi))
            coco["images"].append({
                "id": img_id,
                "file_name": fname,
                "width": PAGE_W,
                "height": PAGE_H
            })
            img_id += 1
            page_global_idx += 1
            if page_global_idx % 10 == 0:
                print(f"Generated {page_global_idx} pages...", end="\r")

    with out_anns.open("w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2, ensure_ascii=False)
    print(f"\nDone. Wrote {len(coco['images'])} images to {out_images}")

def parse_args():
    p = argparse.ArgumentParser(description="Generate synthetic COCO dataset.")
    p.add_argument("--out-images", required=True)
    p.add_argument("--out-anns", required=True)
    p.add_argument("--formulas-file", type=str, default=None)
    p.add_argument("--n-pages", type=int, default=50)
    p.add_argument("--eqs-per-page", type=int, default=4)
    p.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    p.add_argument("--text-size", type=int, default=10, help="Font size (pt) for fake text.")
    p.add_argument("--eq-scale", type=float, default=2.0, help="Scale factor for equations.")
    p.add_argument("--eq-padding", type=int, default=5, help="Padding (px) around equations.")
    p.add_argument("--rotate-aug", action="store_true")
    p.add_argument("--rotate-max", type=float, default=15.0)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--n-papers", type=int, default=5)
    p.add_argument("--debug", action="store_true", help="Disable retries and print render errors.")
    p.add_argument("--emit-recognition-jsonl", action="store_true")
    p.add_argument("--recog-pad", type=int, default=4)
    p.add_argument("--recog-deskew", action="store_true")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.seed is not None:
        import random, numpy as np
        random.seed(args.seed)
        np.random.seed(args.seed)

    generate_synthetic_coco(
        out_images=Path(args.out_images),
        out_anns=Path(args.out_anns),
        n_pages=args.n_pages,
        n_papers=args.n_papers,
        eqs_per_page=args.eqs_per_page,
        dpi=args.dpi,
        seed=args.seed,
        formulas_file=args.formulas_file,
        text_size=args.text_size,
        eq_scale=args.eq_scale,
        eq_padding=args.eq_padding,
        debug_mode=args.debug
    )