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
      --n-pages 50 --eqs-per-page 6 --dpi 150 --rotate-aug --rotate-max 15 --seed 123

Important flags:
    --rotate-aug       enable per-equation rotation augmentation
    --rotate-max       maximum absolute rotation angle in degrees (default 15)
    --seed / --random-seed  reproducible RNG seed for synthetic generation

Dependencies:
    - pdflatex/poppler (optional, used by render_latex)
    - PIL/Pillow, numpy
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
import random
from PIL import ImageDraw, ImageFont
from equation_scribe.config import (
    MAX_EQ_WIDTH_FRAC, MAX_EQ_HEIGHT_FRAC,
    NON_OVERLAP_IOU, MAX_PLACEMENT_ATTEMPTS, ROTATION_AUG_MAX_ANGLE,
    DEFAULT_DPI,PAGE_WIDTH_IN,PAGE_HEIGHT_IN
)

# A small corpus of "science-y" words to generate fake paragraphs
VOCAB = "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore et dolore magna aliqua ut enim ad minim veniam quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur excepteur sint occaecat cupidatat non proident sunt in culpa qui officia deserunt mollit anim id est laborum analysis data system model performance network neural learning algorithm distribution probability vector matrix function linear optimal".split()

# Try to import the repository's render_latex helper (preferred).
# This function should accept (expr: str, out_path: str, dpi: int, prefer_latex: bool)
_render_mathtext = None
try:
    # Attempt absolute import (detector is a package under equation_scribe)
    from equation_scribe.detector.render_latex import render_mathtext as _render_mathtext  # type: ignore
except Exception:
    try:
        # Try local import if run as a script from detector/ directory
        from .render_latex import render_mathtext as _render_mathtext  # type: ignore
    except Exception:
        _render_mathtext = None
try:
    # import the internal latex renderer and the pdf2image flag
    from equation_scribe.detector.render_latex import _latex_render, HAVE_PDF2IMAGE  # type: ignore
except Exception:
    # Not fatal — the code will fall back to matplotlib if pdflatex or pdf2image aren't present
    _latex_render = None
    HAVE_PDF2IMAGE = False

# If render_latex isn't present, we provide a simple matplotlib fallback
if _render_mathtext is None:
    import matplotlib.pyplot as plt
    import matplotlib

    def _matplotlib_render(expr: str, out_path: str, dpi: int = 150, fontsize: int = 20):
        """Fallback renderer: uses matplotlib mathtext to render an expression."""
        # Wrap expression in $...$ if not already math mode (matplotlib expects math mode).
        tex = expr
        if not (tex.startswith("$") and tex.endswith("$")):
            tex = f"${tex}$"

        fig = plt.figure(figsize=(0.01, 0.01))
        fig.text(0.0, 0.0, tex, fontsize=fontsize)
        # Tight bbox to crop around the rendered equation
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)

    def render_mathtext(expr: str, out_path: str, dpi: int = 150, prefer_latex: bool = False):
        # prefer_latex is ignored in the fallback
        return _matplotlib_render(expr, out_path, dpi=dpi, fontsize=10)

    _render_mathtext = render_mathtext  # type: ignore


# Helper: generate a small set of sample LaTeX expressions to render.
SAMPLE_EQUATIONS = [
    r"E = mc^2",
    r"\nabla \cdot \mathbf{E} = \rho/\varepsilon_0",
    r"\int_0^\infty e^{-x^2} dx = \frac{\sqrt{\pi}}{2}",
    r"\frac{d}{dx} \sin x = \cos x",
    r"\begin{pmatrix} a & b \\ c & d \end{pmatrix}",
    r"\sum_{n=1}^\infty \frac{1}{n^2} = \frac{\pi^2}{6}",
    r"\alpha^2 + \beta^2 = \gamma^2",
    r"\mathbf{F} = m \mathbf{a}",
    r"\frac{\partial u}{\partial t} = \nabla^2 u",
    r"\phi(x) = \int K(x,y) f(y) dy",
    r"\lim_{x \to 0} \frac{\sin x}{x} = 1",
]


def clean_latex_formula(tex: str) -> str:
    """
    Sanitizes raw LaTeX using robust logic from validate_linear.py.
    1. Strips comments (handling escaped \% vs \\% correctly).
    2. Strips labels recursively (replacing them with a space).
    """
    if not tex: return ""

    # --- 1. Strip Comments (Robust Backslash Counting) ---
    try:
        idx = 0
        while True:
            # Find next '%'
            idx = tex.index("%", idx)
            
            # Count backslashes immediately preceding it
            backslashes = 0
            i = idx - 1
            while i >= 0 and tex[i] == "\\":
                backslashes += 1
                i -= 1
            
            # Even number of backslashes means the % is NOT escaped -> It's a comment
            if backslashes % 2 == 0:
                tex = tex[:idx]
                break
            else:
                # Odd number means it's escaped (\%), keep searching
                idx += 1
    except ValueError:
        pass # No % found, move on

    # --- 2. Strip Labels (Recursive Brace Counting) ---
    while True:
        # Find start of \label{ (allowing for optional spaces)
        match = re.search(r"\\label\s*\{", tex)
        if not match:
            break
            
        start_idx = match.start()
        open_brace_idx = match.end() - 1
        
        # Walk forward to find the matching closing brace
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
            # Replace with a space to prevent token merging
            tex = tex[:start_idx] + " " + tex[end_idx+1:]
        else:
            # Malformed label. Stop to prevent infinite loop.
            break

    return tex.strip()

def draw_fake_ieee_text(page_img: Image.Image, dpi: int = 150, margin_px: int = 50, col_gap_px: int = 30):
    """
    Draws random text in a two-column layout onto the page_img to simulate an IEEE paper.
    """
    draw = ImageDraw.Draw(page_img)
    w, h = page_img.size
    
    # Define columns
    col_width = (w - (2 * margin_px) - col_gap_px) // 2
    cols = [
        (margin_px, margin_px, margin_px + col_width, h - margin_px), # Left Col
        (margin_px + col_width + col_gap_px, margin_px, w - margin_px, h - margin_px) # Right Col
    ]
    # Calculate pixel size for 10pt font at this DPI
    # IEEE standard is ~10pt for body, ~12-14pt for headers
    body_pt = 10
    header_pt = 12
    
    body_px = int(body_pt * (dpi / 72.0))
    header_px = int(header_pt * (dpi / 72.0))
    
    try:
        # Load font with calculated pixel size
        font = ImageFont.truetype("arial.ttf", body_px) 
        header_font = ImageFont.truetype("arial.ttf", header_px)
    except IOError:
        # Fallback (size might be off, but it prevents crash)
        font = ImageFont.load_default()
        header_font = font

    for (x0, y0, x1, y1) in cols:
        cursor_y = y0
        while cursor_y < y1:
            # Randomly decide if this is a header or paragraph
            is_header = random.random() < 0.1
            current_font = header_font if is_header else font
            
            # Generate a random sentence
            num_words = random.randint(3, 10) if is_header else random.randint(20, 100)
            text = " ".join(random.choices(VOCAB, k=num_words)).capitalize()
            
            # Simple text wrapping logic
            words = text.split()
            line = ""
            for word in words:
                test_line = line + word + " "
                # Check width
                bbox = draw.textbbox((0,0), test_line, font=current_font)
                text_w = bbox[2] - bbox[0]
                
                if text_w < (col_width - 10):
                    line = test_line
                else:
                    draw.text((x0, cursor_y), line, fill="black", font=current_font)
                    line = word + " "
                    cursor_y += 14 if is_header else 12 # Line height
                    
                    if cursor_y > y1: break
            
            # Draw last line
            if cursor_y < y1:
                draw.text((x0, cursor_y), line, fill="black", font=current_font)
                cursor_y += 20 if is_header else 12
            
            # Add paragraph spacing
            cursor_y += random.randint(5, 15)

################################################################################
# stricter placement: try to place boxes with no overlap, optionally fail early
################################################################################
def place_boxes_non_overlapping_strict(
    page_w: int,
    page_h: int,
    box_sizes: List[Tuple[int,int]],
    margin_frac: float = 0.05,
    max_attempts_per_box: int = 1000,
    allow_overlap: bool = False,
) -> List[Tuple[int,int]]:
    """
    Try to place boxes of given sizes (w,h) on a page without overlap. If allow_overlap=False
    the function will *raise* RuntimeError if it cannot place all boxes after the attempts.
    Returns a list of top-left (x,y) placements in the same order as `box_sizes`.

    Args:
      page_w,page_h: page size in pixels
      box_sizes: list of (w,h) for each box to place
      margin_frac: fraction of page width to use as margin (larger margin -> fewer overlaps)
      max_attempts_per_box: attempts per box before failing
      allow_overlap: if True, fall back to placing even if overlaps must occur (backwards compatibility)
    """
    margin = max(1, int(margin_frac * min(page_w, page_h)))  # margin in px
    rects: List[Tuple[int,int,int,int]] = []  # existing placed rectangles (x0,y0,x1,y1)
    placements: List[Tuple[int,int]] = []

    # helper to detect overlap
    def overlaps_any(x0: int, y0: int, x1: int, y1: int) -> bool:
        for ax0,ay0,ax1,ay1 in rects:
            if not (x1 <= ax0 or x0 >= ax1 or y1 <= ay0 or y0 >= ay1):
                return True
        return False

    for idx,(w,h) in enumerate(box_sizes):
        placed_xy = None
        # clamp w,h to page
        w = min(w, page_w - 2*margin)
        h = min(h, page_h - 2*margin)
        if w <= 0 or h <= 0:
            raise RuntimeError(f"Box {idx} is too large for page: w={w} h={h} page=({page_w},{page_h})")
        for attempt in range(max_attempts_per_box):
            x = random.randint(margin, max(margin, page_w - w - margin))
            y = random.randint(margin, max(margin, page_h - h - margin))
            x1 = x + w
            y1 = y + h
            if not overlaps_any(x, y, x1, y1):
                placed_xy = (x,y)
                rects.append((x, y, x1, y1))
                placements.append(placed_xy)
                break
        if placed_xy is None:
            # Could not place without overlap
            if allow_overlap:
                # place at a random location even if overlapping
                x = random.randint(margin, max(margin, page_w - w - margin))
                y = random.randint(margin, max(margin, page_h - h - margin))
                rects.append((x, y, x + w, y + h))
                placements.append((x,y))
            else:
                # fail early with useful diagnostic information
                raise RuntimeError(
                    f"Failed to place box {idx} without overlap after {max_attempts_per_box} attempts. "
                    f"Page size=({page_w},{page_h}), box_size=({w},{h}), margin={margin}."
                )
    return placements



def make_blank_page(width_px: int, height_px: int, color=(255,255,255)) -> Image.Image:
    """Create a blank white page PIL image."""
    return Image.new("RGB", (width_px, height_px), color=color)


def ensure_dirs(*paths: Path):
    """Ensure each path directory exists (if path is a file, ensure parent exists)."""
    for p in paths:
        if p.suffix:  # a file-like path
            p.parent.mkdir(parents=True, exist_ok=True)
        else:
            p.mkdir(parents=True, exist_ok=True)

################################################################################
# helper: compute axis-aligned IoU for two boxes given as (x0,y0,x1,y1)
################################################################################
def compute_iou_xyxy(boxA: Tuple[float,float,float,float], boxB: Tuple[float,float,float,float]) -> float:
    """
    Compute axis-aligned IoU for two boxes in (x0,y0,x1,y1) format.
    Returns IoU in [0,1].
    """
    ax0, ay0, ax1, ay1 = boxA
    bx0, by0, bx1, by1 = boxB
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    areaA = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    areaB = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = areaA + areaB - inter
    if union <= 0:
        return 0.0
    return inter / union

################################################################################
# helper: compute tight axis-aligned bounding box of non-background pixels
################################################################################
def get_tight_bbox(img: Image.Image, bg_thresh: int = 250) -> Optional[Tuple[int,int,int,int]]:
    """
    Return a tight axis-aligned bounding box of non-background pixels for `img`.
    Works for RGBA images (uses alpha) or RGB by thresholding brightness.
    Returns (x0,y0,x1,y1) in img coordinates, or None if image is all background.
    """
    # Ensure RGBA
    if img.mode == "RGBA":
        alpha = img.split()[3]
        bbox = alpha.getbbox()  # (left, upper, right, lower) or None
        if bbox:
            return bbox
        # fall through to intensity check if alpha returned None
    # For RGB / L images: do a brightness threshold to detect non-white
    gray = img.convert("L")
    # threshold: any pixel darker than bg_thresh is considered foreground
    mask = gray.point(lambda p: 255 if p < bg_thresh else 0, mode="L")
    bbox = mask.getbbox()
    return bbox  # may be None


################################################################################
# IoU sanity check for page-level annotations
################################################################################
def assert_no_overlap_page_annotations(page_ann_boxes: List[Tuple[float,float,float,float]], eps: float = 1e-9):
    """
    Given a list of page annotation bboxes (x0,y0,x1,y1), assert that none overlap.
    If any pair has IoU > eps, raises RuntimeError listing the offending pairs.
    """
    n = len(page_ann_boxes)
    bad_pairs = []
    for i in range(n):
        for j in range(i+1, n):
            iou = compute_iou_xyxy(page_ann_boxes[i], page_ann_boxes[j])
            if iou > eps:
                bad_pairs.append((i,j,iou))
    if bad_pairs:
        msg_lines = [f"Found {len(bad_pairs)} overlapping annotation pairs on a page:"]
        for i,j,iou in bad_pairs[:10]:
            msg_lines.append(f"  pair ({i},{j}) IoU={iou:.6f}")
        if len(bad_pairs) > 10:
            msg_lines.append(f"  ... and {len(bad_pairs)-10} more")
        raise RuntimeError("\n".join(msg_lines))
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

def _tight_bbox_from_rgba(img_rgba: Image.Image):
    """Return tight bbox (x0,y0,x1,y1) of non-transparent alpha in an RGBA image."""
    if img_rgba.mode != "RGBA":
        return (0, 0, img_rgba.width, img_rgba.height)
    alpha = img_rgba.split()[3]
    bbox = alpha.getbbox()  # returns None or (left, upper, right, lower)
    if bbox is None:
        return (0, 0, img_rgba.width, img_rgba.height)
    return bbox

def place_and_annotate_on_page(
    page_img: Image.Image,
    eq_images: list,
    page_annotations: list,
    rotate_aug: bool = False,
    rotate_max: float = 15.0,
    require_non_overlap: bool = True,
    margin_frac: float = 0.05,
    max_attempts_per_box: int = 1000,
):
    """
    Place rendered equation images onto a page, ensuring non-overlap (optional),
    cropping rotated images tightly and producing annotations.

    Args:
        page_img (PIL.Image.Image): page image to paste equations onto (modified in-place).
        eq_images (List[Tuple[str, PIL.Image.Image]]): list of (latex_str, image).
        page_annotations (List[dict]): list to append annotations to. Each annotation is
            appended as {"latex": str, "bbox": [x0,y0,x1,y1], "angle": float}.
        rotate_aug (bool): whether to randomly rotate each equation image.
        rotate_max (float): maximum absolute rotation angle in degrees.
        require_non_overlap (bool): if True, attempts to place boxes with IoU <= NON_OVERLAP_IOU.
        margin_frac (float): page margin fraction (of page width) to avoid placing near edges.
        max_attempts_per_box (int): maximum attempts to find a valid placement for a box.

    Returns:
        None

    Notes:
        - The function pastes equations (composited on white) onto `page_img` and
          appends annotation dicts into `page_annotations`.
        - Coordinates in the returned bbox are image pixel coordinates (x0,y0,x1,y1).
    """
        
    PAGE_W, PAGE_H = page_img.size
    margin = int(round(margin_frac * PAGE_W))
    placed_boxes = []

    for latex, eq_img in eq_images:
        # eq_img should be RGBA
        if eq_img.mode != "RGBA":
            eq_img = eq_img.convert("RGBA")

        # rotation
        angle = random.uniform(-rotate_max, rotate_max) if rotate_aug else 0.0
        if abs(angle) > 1e-6:
            rotated = eq_img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=(255,255,255,0))
        else:
            rotated = eq_img

        tx0, ty0, tx1, ty1 = _tight_bbox_from_rgba(rotated)
        cropped = rotated.crop((tx0, ty0, tx1, ty1))
        w, h = cropped.size

        # Enforce reasonable size (page-relative)
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
            if cand[2] > PAGE_W - margin or cand[3] > PAGE_H - margin:
                continue

            if require_non_overlap:
                overlap = False
                for prev in placed_boxes:
                    if _iou(cand, prev) > NON_OVERLAP_IOU:
                        overlap = True
                        break
                if overlap:
                    continue

            # Paste: composite alpha over white background to keep consistent page look
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
            # fallback: try to find best minimal overlap spot
            best_spot = None
            best_max_iou = float("inf")
            for _ in range(200):
                x = random.randint(margin, max(margin, PAGE_W - margin - w))
                y = random.randint(margin, max(margin, PAGE_H - margin - h))
                cand = (x, y, x + w, y + h)
                max_iou = max((_iou(cand, prev) for prev in placed_boxes), default=0.0)
                if max_iou < best_max_iou:
                    best_max_iou = max_iou
                    best_spot = (x, y, cand)
            if best_spot:
                x, y, cand = best_spot
                if cropped.mode == "RGBA":
                    bg = Image.new("RGB", (w, h), (255, 255, 255))
                    bg.paste(cropped, mask=cropped.split()[3])
                    page_img.paste(bg, (x, y))
                else:
                    page_img.paste(cropped.convert("RGB"), (x, y))
                page_annotations.append({"latex": latex, "bbox": [float(x), float(y), float(x + w), float(y + h)], "angle": float(angle)})
                placed_boxes.append(cand)
            else:
                # Could not place; skip this equation (log for debug)
                logger = logging.getLogger(__name__)
                logger.warning("Could not place equation (skipping): %s", latex)
                continue

def generate_synthetic_coco(out_images: Path, out_anns: Path,
                            n_pages: int = 50,
                            n_papers: int = 5,
                            eqs_per_page: int = 4,
                            dpi: int = DEFAULT_DPI,
                            seed: int = 0,
                            formulas_file: Optional[Path] = None,):
    """
    Generate synthetic pages and COCO annotations.

    Args:
        out_images (Path): directory where generated images will be saved.
        out_anns (Path): path to the COCO annotations JSON file to write.
        n_pages (int): number of pages to generate.
        eqs_per_page (int): number of equations per page.
        dpi (int): DPI used for equation rendering and saved images.
        rotate_aug (bool): whether to apply per-equation rotation.
        rotate_max (float): maximum absolute angle (degrees) to rotate equations by.
        seed (Optional[int]): random seed for reproducible generation.
    Returns:
        None

    Side effects:
        Writes image files to `out_images` and a COCO JSON file to `out_anns`.
    """
    random.seed(seed)

    PAGE_W = int(PAGE_WIDTH_IN * dpi)
    PAGE_H = int(PAGE_HEIGHT_IN * dpi)

    out_images = Path(out_images)
    out_anns = Path(out_anns)
    out_images.mkdir(parents=True, exist_ok=True)
    out_anns.parent.mkdir(parents=True, exist_ok=True)

    # Distribute pages across papers (roughly equal)
    pages_per_paper = [n_pages // n_papers] * n_papers
    for i in range(n_pages % n_papers):
        pages_per_paper[i] += 1

    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "equation"}],
    }

    img_id = 1
    ann_id = 1
    paper_idx = 0
    page_global_idx = 0

    # 1. LOAD FORMULAS
    # all_formulas = SAMPLE_EQUATIONS # Fallback
    
    if formulas_file and Path(formulas_file).exists():
        p_formulas = Path(formulas_file)
        print(f"Loading formulas from {p_formulas}...")
        
        # Robust Open: Try UTF-8 first, fallback to Latin-1 (common in older datasets)
        try:
            with open(p_formulas, "r", encoding="utf-8") as f:
                raw_lines = f.readlines()
        except UnicodeDecodeError:
            print("UTF-8 failed, trying latin-1...")
            with open(p_formulas, "r", encoding="latin-1") as f:
                raw_lines = f.readlines()

        all_formulas = []
        if p_formulas.suffix.lower() == ".csv":
            # Keep legacy CSV support
            import csv
            reader = csv.reader(raw_lines)
            for row in reader:
                if row:
                    clean = clean_latex_formula(row[0])
                    if len(clean) > 5:
                        all_formulas.append(clean)
        else:
            # LST Mode: Apply robust cleaner
            for line in raw_lines:
                clean = clean_latex_formula(line.strip())
                if len(clean) > 5:
                    all_formulas.append(clean)
                
        print(f"Loaded {len(all_formulas)} clean formulas.")

    for paper_idx in range(n_papers):
        pages_for_this = pages_per_paper[paper_idx]
        for page_idx in range(pages_for_this):
            # file name includes paper id and page index
            fname = f"paper{paper_idx:03d}_page_{page_global_idx:04d}.png"
            fpath = out_images / fname

            page_img = make_blank_page(PAGE_W, PAGE_H)
            # Generate eq image sizes by rendering each equation first to a temp file,
            # then measure size and paste onto the page.
            # 2. NEW: Draw the IEEE-style text first
            draw_fake_ieee_text(page_img, dpi=dpi) 
            
            # Select random formulas
            eq_exprs = [random.choice(all_formulas) for _ in range(eqs_per_page)]
            eq_images = []
            eq_sizes = []
            for expr in eq_exprs:
                 # print(expr)
                # Render using repository's renderer (or fallback)
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpf:
                    tmpname = tmpf.name
                try:
                    _render_mathtext(expr, tmpname, dpi=dpi, prefer_latex=True)
                    eq_img = Image.open(tmpname)
                    # Force load into memory and close file descriptor so we can unlink tmp file safely
                    eq_img.load()
                    # Convert to RGBA for rotation/compositing convenience
                    eq_img = eq_img.convert("RGBA")
                    # --- NEW: Hard correction for size mismatch ---
                    # If equations are 2-3x too big, scale them down here.
                    # 0.5 is a good starting point if they are "double size".
                    # scale_correction = 0.5 
                    # new_w = max(1, int(eq_img.width * scale_correction))
                    # new_h = max(1, int(eq_img.height * scale_correction))
                    # eq_img = eq_img.resize((new_w, new_h), Image.LANCZOS)
                except Exception:
                    # Fallback: draw a small placeholder box with the expression text
                    eq_img = Image.new("RGBA", (int(dpi * 0.5), int(dpi * 0.2)), (200, 200, 200, 255))
                finally:
                    try:
                        os.unlink(tmpname)
                    except Exception:
                        # If unlink fails, leave the temp to inspect later (or log)
                        logger = logging.getLogger(__name__)
                        logger.debug("Could not unlink temp file %s", tmpname)


                if eq_img is None:
                    # Defensive fallback
                    eq_img = Image.new("RGBA", (int(dpi * 0.5), int(dpi * 0.2)), (200, 200, 200, 255))


                # At this point eq_img should be a PIL.Image (RGBA)
                if eq_img is None:
                    # defensive: create a visible placeholder if something odd happened
                    eq_img = Image.new("RGBA", (int(dpi * 0.5), int(dpi * 0.2)), (200, 200, 200, 255))

                # Ensure a reasonable size - enforce max width/height relative to page
                max_w = int(PAGE_W * 0.9)
                max_h = int(PAGE_H * 0.25)
                w, h = eq_img.size
                if w > max_w or h > max_h:
                    scale = min(max_w / w, max_h / h)
                    new_w = max(10, int(w * scale))
                    new_h = max(10, int(h * scale))
                    eq_img = eq_img.resize((new_w, new_h), Image.LANCZOS)

                eq_images.append((expr, eq_img))
                eq_sizes.append(eq_img.size)

            # Use the centralized placement helper which handles rotation, tight bbox,
            # IoU checks, cropping and pasting. It appends entries to page_records.
            page_records = []
            place_and_annotate_on_page(
                page_img=page_img,
                eq_images=eq_images,
                page_annotations=page_records,
                rotate_aug=args.rotate_aug,
                rotate_max=args.rotate_max,
                require_non_overlap=True,
                margin_frac=0.05,
                max_attempts_per_box=1000,
            )

            # Append page_records into COCO annotations (maintain ann_id)
            for rec in page_records:
                x0, y0, x1, y1 = rec["bbox"]
                w = int(round(x1 - x0))
                h = int(round(y1 - y0))
                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 1,
                    "bbox": [int(round(x0)), int(round(y0)), w, h],
                    "area": int(w * h),
                    "iscrowd": 0,
                    "paper_id": f"paper{paper_idx:03d}",
                    "page_index": page_idx,
                    "latex": rec["latex"],
                    "angle": rec.get("angle", 0.0),
                })
                ann_id += 1
            # Strict placement and annotation (handles rotated tight bboxes and IoU checks)
            page_records = []  # will accumulate page-level annotations (latex + bbox)

            # Save the page image
            page_img.save(fpath, format="PNG", dpi=(dpi, dpi))

            coco["images"].append({
                "id": img_id,
                "file_name": fname,
                "width": PAGE_W,
                "height": PAGE_H,
                "paper_id": f"paper{paper_idx:03d}",
                "page_index": page_idx,
            })
            img_id += 1
            page_global_idx += 1

    # Write COCO annotations
    with out_anns.open("w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2, ensure_ascii=False)
    # optional: emit recognition pairs automatically (if requested via CLI flag)
    if args.emit_recognition_jsonl:
        try:
            from equation_scribe.detector.make_recognition_pairs import build_pairs_from_coco
            pairs_out_dir = Path(out_images.parent) / "recognition_pairs" / "crops"
            pairs_jsonl = Path(out_images.parent) / "recognition_pairs" / Path(out_anns).name.replace("instances_all.json", "recognition_all.jsonl")
            build_pairs_from_coco(out_anns, out_images, pairs_out_dir, pairs_jsonl, pad_px=args.recog_pad, deskew=args.recog_deskew)
            print(f"Emitted recognition JSONL to {pairs_jsonl}")
        except Exception as e:
            print(f"WARNING: failed to emit recognition pairs: {e}")


    print(f"Wrote {len(coco['images'])} images and {len(coco['annotations'])} annotations.")
    print(f"Images directory: {out_images.resolve()}")
    print(f"COCO annotations: {out_anns.resolve()}")


def parse_args():
    p = argparse.ArgumentParser(description="Generate synthetic multi-paper COCO dataset.")
    p.add_argument("--out-images", required=True, help="Directory to write page PNG images.")
    p.add_argument("--out-anns", required=True, help="Path to write COCO JSON (instances_all.json).")
    p.add_argument("--n-pages", type=int, default=50, help="Total number of pages across all papers.")
    p.add_argument("--n-papers", type=int, default=5, help="Number of distinct synthetic papers.")
    p.add_argument("--eqs-per-page", type=int, default=4, help="Number of equations per page.")
    p.add_argument("--dpi", type=int, default=DEFAULT_DPI, help="DPI for rendered pages.")
    p.add_argument("--rotate", action="store_true", help="Enable rotation augmentation for equation renderings")
    p.add_argument("--rotate-aug", action="store_true",
                    help="Enable per-equation rotation augmentation when placing equations.")
    p.add_argument("--rotate-max", type=float, default=15.0,
                        help="Maximum absolute rotation angle (degrees) for per-equation augmentation.")
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducible synthetic generation.")
    p.add_argument("--emit-recognition-jsonl", action="store_true", help="Emit recognition JSONL/crops after generating COCO")
    p.add_argument("--recog-pad", type=int, default=4, help="Padding in px for recognition crop")
    p.add_argument("--recog-deskew", action="store_true", help="Attempt to deskew recognition crops")
    p.add_argument("--formulas-file", type=str, default=None, 
                   help="Path to a text file (.lst) or CSV (.csv) containing LaTeX formulas.")


    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ### SET RANDOM SEED IF PROVIDED ###
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
    )
