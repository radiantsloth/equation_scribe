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
* The script attempts to use the local `detector.render_latex.render_mathtext`
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
from typing import List, Tuple

from PIL import Image

# Try to import the repository's render_latex helper (preferred).
# This function should accept (expr: str, out_path: str, dpi: int, prefer_latex: bool)
_render_mathtext = None
try:
    # Attempt absolute import (detector is a package under equation_scribe)
    from detector.render_latex import render_mathtext as _render_mathtext  # type: ignore
except Exception:
    try:
        # Try local import if run as a script from detector/ directory
        from .render_latex import render_mathtext as _render_mathtext  # type: ignore
    except Exception:
        _render_mathtext = None

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


def place_boxes_non_overlapping(page_w: int, page_h: int, 
                                 box_sizes: List[Tuple[int,int]],
                                 margin: int = 20,
                                 max_attempts: int = 200) -> List[Tuple[int,int]]:
    """
    Given a page size and a list of box widths/heights, attempt to place each
    box onto the page without overlapping previously placed boxes.

    Returns a list of (x, y) top-left coordinates for each box in box_sizes order.

    This is a greedy randomized algorithm that attempts up to max_attempts per box.
    """
    placed = []
    rects = []  # list of (x0, y0, x1, y1)

    for (w, h) in box_sizes:
        placed_xy = None
        for attempt in range(max_attempts):
            x = random.randint(margin, max(0, page_w - w - margin))
            y = random.randint(margin, max(0, page_h - h - margin))
            x1, y1 = x + w, y + h
            overlap = False
            for (ax0, ay0, ax1, ay1) in rects:
                # check overlap
                if not (x1 <= ax0 or x >= ax1 or y1 <= ay0 or y >= ay1):
                    overlap = True
                    break
            if not overlap:
                placed_xy = (x, y)
                rects.append((x, y, x1, y1))
                break
        if placed_xy is None:
            # give up and place at random possibly overlapping position
            x = max(margin, min(page_w - w - margin, random.randint(margin, page_w - w - margin)))
            y = max(margin, min(page_h - h - margin, random.randint(margin, page_h - h - margin)))
            placed_xy = (x, y)
            rects.append((x, y, x + w, y + h))
        placed.append(placed_xy)
    return placed


def make_blank_page(width_px: int, height_px: int, color=(255,255,255)) -> Image.Image:
    """Create a blank white page PIL image."""
    return Image.new("RGB", (width_px, height_px), color=color)


def ensure_dirs(*paths: Path):
    for p in paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.mkdir(parents=True, exist_ok=True) if p.suffix == "" else p.parent.mkdir(parents=True, exist_ok=True)


def generate_synthetic_coco(out_images: Path, out_anns: Path,
                            n_pages: int = 50,
                            n_papers: int = 5,
                            eqs_per_page: int = 4,
                            dpi: int = DEFAULT_DPI,
                            seed: int = 0):
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
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpf:
                    tmpname = tmpf.name
                try:
                    # Render using repository's renderer (or fallback)
                    _render_mathtext(expr, tmpname, dpi=dpi, prefer_latex=True)
                    eq_img = Image.open(tmpname).convert("RGBA")
                except Exception:
                    # Fallback: draw a small placeholder box with the expression text
                    eq_img = Image.new("RGBA", (int(dpi*0.5), int(dpi*0.2)), (200, 200, 200, 255))
                finally:
                    # Try to clean up the temp file (Image sometimes keeps it open)
                    try:
                        os.unlink(tmpname)
                    except Exception:
                        pass

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

            # Place boxes non-overlapping
            placements = place_boxes_non_overlapping(PAGE_W, PAGE_H, eq_sizes, margin=int(0.05*PAGE_W))

            # Paste the equation images and create COCO annotations
            for (expr, eq_img), (x, y) in zip(eq_images, placements):
                # If eq_img has alpha channel, composite it against white
                if eq_img.mode == "RGBA":
                    bg = Image.new("RGB", eq_img.size, (255,255,255))
                    bg.paste(eq_img, mask=eq_img.split()[3])
                    paste_img = bg
                else:
                    paste_img = eq_img.convert("RGB")

                page_img.paste(paste_img, (x, y))

                w, h = paste_img.size
                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 1,
                    "bbox": [int(x), int(y), int(w), int(h)],
                    "area": int(w * h),
                    "iscrowd": 0,
                    # optional extras for downstream convenience
                    "paper_id": f"paper{paper_idx:03d}",
                    "page_index": page_idx,
                    "latex": expr,
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
