#!/usr/bin/env python3
"""
synthetic_coco.py

Generate a synthetic COCO-style annotations file and synthetic page images,
organized into multiple "papers" so that downstream split-by-paper tooling
can produce both train and val splits.

Usage (from repo root):
  python equation_scribe/detector/synthetic_coco.py \
    --out-images detector/data/images/synth \
    --out-anns detector/data/annotations/instances_all.json \
    --n-pages 50 \
    --n-papers 5 \
    --eqs-per-page 4 \
    --dpi 150

Notes:
* Images will be named like `paper000_page_0000.png`, etc.  This is required
  by split_coco_by_paper.py so it can detect which pages belong to the same
  paper.
* The script attempts to use the local `equation_scribe.detector.render_latex.render_mathtext`
  renderer if available (that renderer uses LaTeX or matplotlib).  If that's
  unavailable, a matplotlib-based fallback renderer is used.
"""

from __future__ import annotations
import argparse
import json
import math
import os
import random
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional
import shutil

from PIL import Image

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
        return _matplotlib_render(expr, out_path, dpi=dpi, fontsize=22)

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

# Basic page template constants
PAGE_WIDTH_IN = 8.5   # inches (letter)
PAGE_HEIGHT_IN = 11.0 # inches
DEFAULT_DPI = 150


# def place_boxes_non_overlapping(page_w: int, page_h: int, 
    #                              box_sizes: List[Tuple[int,int]],
    #                              margin: int = 20,
    #                              max_attempts: int = 200) -> List[Tuple[int,int]]:
    # """
    # Given a page size and a list of box widths/heights, attempt to place each
    # box onto the page without overlapping previously placed boxes.

    # Returns a list of (x, y) top-left coordinates for each box in box_sizes order.

    # This is a greedy randomized algorithm that attempts up to max_attempts per box.
    # """
    # placed = []
    # rects = []  # list of (x0, y0, x1, y1)

    # for (w, h) in box_sizes:
    #     placed_xy = None
    #     for attempt in range(max_attempts):
    #         x = random.randint(margin, max(0, page_w - w - margin))
    #         y = random.randint(margin, max(0, page_h - h - margin))
    #         x1, y1 = x + w, y + h
    #         overlap = False
    #         for (ax0, ay0, ax1, ay1) in rects:
    #             # check overlap
    #             if not (x1 <= ax0 or x >= ax1 or y1 <= ay0 or y >= ay1):
    #                 overlap = True
    #                 break
    #         if not overlap:
    #             placed_xy = (x, y)
    #             rects.append((x, y, x1, y1))
    #             break
    #     if placed_xy is None:
    #         # give up and place at random possibly overlapping position
    #         x = max(margin, min(page_w - w - margin, random.randint(margin, page_w - w - margin)))
    #         y = max(margin, min(page_h - h - margin, random.randint(margin, page_h - h - margin)))
    #         placed_xy = (x, y)
    #         rects.append((x, y, x + w, y + h))
    #     placed.append(placed_xy)
    # return placed
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
    for p in paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.mkdir(parents=True, exist_ok=True) if p.suffix == "" else p.parent.mkdir(parents=True, exist_ok=True)

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

def place_and_annotate_on_page(
    page_img: Image.Image,
    eq_images: List[Tuple[str, Image.Image]],
    page_annotations: list,
    require_non_overlap: bool = True,
    margin_frac: float = 0.05,
    max_attempts_per_box: int = 1000,
    rotate: bool = False,
):
    """
    Places eq_images (list of (latex_str, PIL.Image)) on page_img non-overlapping,
    crops tight bbox after any rotation, pastes the cropped image, and appends
    page_annotations entries with bbox in page coords [x0,y0,x1,y1].

    Raises RuntimeError if cannot place without overlap (when require_non_overlap=True).
    """
    PAGE_W, PAGE_H = page_img.size

    # compute tight sizes for each eq image
    box_sizes = []
    processed = []
    for expr, img in eq_images:
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        # If you want rotations, rotate here, e.g.:
        if rotate:
            angle = random.uniform(-10, 10)
            img = img.rotate(angle, expand=True)

        tight = get_tight_bbox(img, bg_thresh=250)
        if tight is None:
            tw, th = img.size
            tight = (0, 0, tw, th)
        tx0, ty0, tx1, ty1 = tight
        tw = max(1, int(round(tx1 - tx0)))
        th = max(1, int(round(ty1 - ty0)))
        box_sizes.append((tw, th))
        processed.append((expr, img, tight))

    # strict placement
    placements = place_boxes_non_overlapping_strict(
        PAGE_W, PAGE_H, box_sizes,
        margin_frac=margin_frac, max_attempts_per_box=max_attempts_per_box, allow_overlap=False
    )

    # paste and create annotations
    page_boxes = []
    for (expr, img, tight), (x, y) in zip(processed, placements):
        tx0, ty0, tx1, ty1 = tight
        cropped = img.crop((tx0, ty0, tx1, ty1)).convert("RGB")
        page_img.paste(cropped, (x, y))
        px0 = float(x); py0 = float(y)
        px1 = float(x + (tx1 - tx0)); py1 = float(y + (ty1 - ty0))
        page_boxes.append((px0, py0, px1, py1))
        page_annotations.append({"latex": expr, "bbox": [px0, py0, px1, py1], "type": "display"})

    if require_non_overlap:
        assert_no_overlap_page_annotations(page_boxes, eps=1e-9)

    return page_annotations

def generate_synthetic_coco(out_images: Path, out_anns: Path,
                            n_pages: int = 50,
                            n_papers: int = 5,
                            eqs_per_page: int = 4,
                            dpi: int = DEFAULT_DPI,
                            seed: int = 0,):
    """
    Generate synthetic pages and a COCO-style annotations JSON file containing
    the synthetic equation boxes.

    out_images: directory to write page images (PNG)
    out_anns:  path to write JSON (COCO) annotations
    n_pages:  total number of pages across all papers
    n_papers: number of papers to split the pages into
    eqs_per_page: number of synthetic equations to place per page
    dpi:     dots per inch for image generation
    seed:    random seed for repeatability
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

    for paper_idx in range(n_papers):
        pages_for_this = pages_per_paper[paper_idx]
        for page_idx in range(pages_for_this):
            # file name includes paper id and page index
            fname = f"paper{paper_idx:03d}_page_{page_global_idx:04d}.png"
            fpath = out_images / fname

            page_img = make_blank_page(PAGE_W, PAGE_H)
            # Generate eq image sizes by rendering each equation first to a temp file,
            # then measure size and paste onto the page.
            eq_exprs = [random.choice(SAMPLE_EQUATIONS) for _ in range(eqs_per_page)]
            eq_images = []
            eq_sizes = []
            for expr in eq_exprs:
                                # render into a temporary PNG (tmpname)
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpf:
                    tmpname = tmpf.name

                eq_img = None

                # DEBUG: print LaTeX availability so logs make it clear
                # (this will help in diagnosing intermittent environment/path differences)
                if _latex_render is None or shutil.which("pdflatex") is None or not HAVE_PDF2IMAGE:
                    print(f"[DEBUG] pdflatex available? {shutil.which('pdflatex') is not None}; pdf2image available? {HAVE_PDF2IMAGE}")

                # If the expression looks like a LaTeX environment, force the LaTeX route
                if "\\begin" in expr and _latex_render is not None and shutil.which("pdflatex") and HAVE_PDF2IMAGE:
                    try:
                        # Call the real LaTeX renderer directly (bypass matplotlib entirely)
                        _latex_render(expr, tmpname, dpi=max(dpi, 300))
                        eq_img = Image.open(tmpname).convert("RGBA")
                    except Exception as latex_exc:
                        # If the real LaTeX render fails here, print detailed diagnostics
                        print("----------------------------------------------------------------------")
                        print("[ERROR] _latex_render failed for expression:", repr(expr))
                        print("[ERROR] Exception:", latex_exc)
                        # attempt matplotlib fallback (it will fail for \\begin{...}, but try anyway)
                        try:
                            _render_mathtext(expr, tmpname, dpi=dpi, prefer_latex=False)
                            eq_img = Image.open(tmpname).convert("RGBA")
                            print("[INFO] matplotlib fallback succeeded for expression:", repr(expr))
                        except Exception as mpl_exc:
                            print("[ERROR] matplotlib fallback also failed for expression:", repr(expr))
                            print(" LaTeX exception:", latex_exc)
                            print(" Matplotlib exception:", mpl_exc)
                            # print any nearby .tex/.log files for quick debugging if present
                            try:
                                tmpdir = Path(tmpname).parent
                                for tf in tmpdir.glob("*.log")[:3]:
                                    print(f" --- contents of {tf.name} ---")
                                    print(tf.read_text(errors="ignore")[:2000])
                            except Exception:
                                pass
                            eq_img = Image.new("RGBA", (int(dpi * 0.5), int(dpi * 0.2)), (200, 200, 200, 255))
                else:
                    # Non-environment expressions: use the regular renderer which will try LaTeX
                    # when appropriate (and fall back to matplotlib).
                    try:
                        _render_mathtext(expr, tmpname, dpi=dpi, prefer_latex=True)
                        eq_img = Image.open(tmpname).convert("RGBA")
                    except Exception as e_first:
                        # Try fallback (matplotlib)
                        print("[WARN] render_mathtext(prefer_latex=True) failed; trying matplotlib fallback.", repr(expr))
                        try:
                            _render_mathtext(expr, tmpname, dpi=dpi, prefer_latex=False)
                            eq_img = Image.open(tmpname).convert("RGBA")
                            print("[INFO] matplotlib fallback succeeded for expression:", repr(expr))
                        except Exception as e_second:
                            print("[ERROR] Both LaTeX and matplotlib rendering failed for:", repr(expr))
                            print(" First exception:", e_first)
                            print(" Second exception:", e_second)
                            eq_img = Image.new("RGBA", (int(dpi * 0.5), int(dpi * 0.2)), (200, 200, 200, 255))
                # finally: clean up temp file if present
                try:
                    if Path(tmpname).exists():
                        Path(tmpname).unlink()
                except Exception:
                    pass

                if eq_img is None:
                    # Defensive fallback
                    eq_img = Image.new("RGBA", (int(dpi * 0.5), int(dpi * 0.2)), (200, 200, 200, 255))


                # At this point eq_img should be a PIL.Image (RGBA)
                if eq_img is None:
                    # defensive: create a visible placeholder if something odd happened
                    eq_img = Image.new("RGBA", (int(dpi * 0.5), int(dpi * 0.2)), (200, 200, 200, 255))

                # with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpf:
                #     tmpname = tmpf.name
                # try:
                #     # Render using repository's renderer (or fallback)
                #     _render_mathtext(expr, tmpname, dpi=dpi, prefer_latex=True)
                #     # render_mathtext(expr, tmpname, dpi=dpi)
                #     eq_img = Image.open(tmpname).convert("RGBA")
                # except Exception:
                #     # Fallback: draw a small placeholder box with the expression text
                #     eq_img = Image.new("RGBA", (int(dpi*0.5), int(dpi*0.2)), (200, 200, 200, 255))
                # finally:
                #     # Try to clean up the temp file (Image sometimes keeps it open)
                #     try:
                #         os.unlink(tmpname)
                #     except Exception:
                #         pass

                # Ensure a reasonable size - enforce max width/height relative to page
                max_w = int(PAGE_W * 0.6)
                max_h = int(PAGE_H * 0.25)
                w, h = eq_img.size
                if w > max_w or h > max_h:
                    scale = min(max_w / w, max_h / h)
                    new_w = max(10, int(w * scale))
                    new_h = max(10, int(h * scale))
                    eq_img = eq_img.resize((new_w, new_h), Image.LANCZOS)

                eq_images.append((expr, eq_img))
                eq_sizes.append(eq_img.size)

            # # Place boxes non-overlapping
            # placements = place_boxes_non_overlapping(PAGE_W, PAGE_H, eq_sizes, margin=int(0.05*PAGE_W))

            # # Paste the equation images and create COCO annotations
            # for (expr, eq_img), (x, y) in zip(eq_images, placements):
            #     # If eq_img has alpha channel, composite it against white
            #     if eq_img.mode == "RGBA":
            #         bg = Image.new("RGB", eq_img.size, (255,255,255))
            #         bg.paste(eq_img, mask=eq_img.split()[3])
            #         paste_img = bg
            #     else:
            #         paste_img = eq_img.convert("RGB")

            #     page_img.paste(paste_img, (x, y))

            #     w, h = paste_img.size
            #     coco["annotations"].append({
            #         "id": ann_id,
            #         "image_id": img_id,
            #         "category_id": 1,
            #         "bbox": [int(x), int(y), int(w), int(h)],
            #         "area": int(w * h),
            #         "iscrowd": 0,
            #         # optional extras for downstream convenience
            #         "paper_id": f"paper{paper_idx:03d}",
            #         "page_index": page_idx,
            #         "latex": expr,
            #     })
            #     ann_id += 1
            # Strict placement and annotation (handles rotated tight bboxes and IoU checks)
            page_records = []  # will accumulate page-level annotations (latex + bbox)
            # call our helper to place, paste, and populate page_records
            place_and_annotate_on_page(
                page_img=page_img,
                eq_images=eq_images,
                page_annotations=page_records,
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
                })
                ann_id += 1

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
    p.add_argument("--seed", type=int, default=0, help="Random seed.")
    p.add_argument("--rotate", action="store_true", help="Enable rotation augmentation for equation renderings")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_synthetic_coco(
        out_images=Path(args.out_images),
        out_anns=Path(args.out_anns),
        n_pages=args.n_pages,
        n_papers=args.n_papers,
        eqs_per_page=args.eqs_per_page,
        dpi=args.dpi,
        seed=args.seed,
    )
