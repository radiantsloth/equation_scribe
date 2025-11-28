#!/usr/bin/env python3
"""
render_latex.py

Render LaTeX math expressions to PNG images.

- For simple inline math, Matplotlib's mathtext is used (fast).
- For expressions that use LaTeX environments like \begin{pmatrix}, the script
  falls back to running pdflatex (real LaTeX engine) and converting the PDF
  to PNG via pdf2image.

Dependencies:
- matplotlib (for simple mathtext)
- pdf2image (for LaTeX route) and poppler (system)
- A LaTeX engine (pdflatex) for full LaTeX rendering (MikTeX or TeX Live)
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
import argparse
import sys
import logging

# matplotlib path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# pdf2image optional
try:
    from pdf2image import convert_from_path
    HAVE_PDF2IMAGE = True
except Exception:
    HAVE_PDF2IMAGE = False

logger = logging.getLogger("render_latex")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
logger.addHandler(ch)


def _matplotlib_render(expr: str, out_path: str, dpi: int = 200, fontsize: int = 28):
    """Render the expression with matplotlib mathtext (fast, but limited)."""
    fig = plt.figure(figsize=(3, 1))
    # place in figure center
    plt.text(0.5, 0.5, f"${expr}$", fontsize=fontsize, ha="center", va="center")
    plt.axis("off")
    # save to a temporary file and then re-open to ensure RGB
    tmp = out_path + ".tmp.png"
    fig.savefig(tmp, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    # convert to RGB to normalize format
    Image.open(tmp).convert("RGB").save(out_path)
    try:
        os.remove(tmp)
    except Exception:
        pass
    return out_path


def _latex_render(expr: str, out_path: str, dpi: int = 300, packages=None):
    """
    Render expression using pdflatex -> pdf -> png.

    Uses the standalone documentclass (tight bounding) and ensures the expression
    is placed inside math mode when appropriate (especially for \begin{...}).
    """
    if shutil.which("pdflatex") is None:
        raise RuntimeError("pdflatex not found on PATH. Please install TeX (MiKTeX or TeX Live).")

    if not HAVE_PDF2IMAGE:
        raise RuntimeError("pdf2image not installed; needed to convert PDF -> PNG. `pip install pdf2image` and install poppler.")

    packages = packages or ["amsmath", "amssymb", "amsfonts", "bm"]

    s = expr.strip()

    # Decide whether this needs display-math wrapping.
    # If it contains a LaTeX environment (\begin{...}) or multiline content, use display math.
    needs_display_math = False
    if "\\begin" in s or "\\cases" in s or "\\align" in s or "\n" in s:
        needs_display_math = True

    # Detect if expression already explicitly uses math delimiters ($, \(, \[, or display environments).
    already_math = False
    if s.startswith("$") or s.startswith("\\(") or s.startswith("\\["):
        already_math = True
    # Note: do NOT treat "\begin{" as already math — we want to wrap that case.

    # Construct the math block to place inside the standalone document.
    if needs_display_math and not already_math:
        math_block = "\\[\n" + expr + "\n\\]"
    elif not needs_display_math and not already_math:
        # Use inline math for short expressions
        math_block = "\\(" + expr + "\\)"
    else:
        # Already has math delimiters or is intentionally a LaTeX fragment
        math_block = expr

    tex = r"""\documentclass[varwidth=true, border=2pt]{standalone}
\usepackage{%s}
\begin{document}
%s
\end{document}
""" % (",".join(packages), math_block)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        tex_file = td / "eq.tex"
        tex_file.write_text(tex, encoding="utf-8")

        # run pdflatex
        cmd = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "-output-directory", str(td), str(tex_file)]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            # capture log
            log = (td / "eq.log").read_text(errors="ignore") if (td / "eq.log").exists() else ""
            raise RuntimeError(f"pdflatex failed: {e}\nlog:\n{log}") from e

        pdf_file = td / "eq.pdf"
        if not pdf_file.exists():
            raise RuntimeError("pdflatex did not produce eq.pdf")

        # convert to PNG using pdf2image
        pages = convert_from_path(str(pdf_file), dpi=dpi, fmt="png")
        if len(pages) == 0:
            raise RuntimeError("pdf2image did not return any pages")
        pages[0].convert("RGB").save(out_path)
    return out_path




def render_mathtext(expr: str, out_path: str, dpi: int = 200, fontsize: int = 28, prefer_latex: bool = False):
    """
    Render a LaTeX expression to out_path (PNG).

    - expr: LaTeX expression (e.g., '\\nabla \\cdot E = \\rho/\\varepsilon_0' or '\\begin{pmatrix} ...')
    - out_path: target png path
    - dpi: resolution for rendering
    - fontsize: used for matplotlib route
    - prefer_latex: if True, always try pdflatex route; otherwise autodetect
    """
    out_path = str(out_path)
    # heuristics: if expression contains a LaTeX environment, or multi-line constructs, use pdflatex
    needs_full_latex = "\\begin" in expr or "\\matrix" in expr or "\\begin{" in expr or "\n" in expr or "\\displaystyle" in expr or "\\cases" in expr or "\\align" in expr

    if prefer_latex or needs_full_latex:
        try:
            return _latex_render(expr, out_path, dpi=max(dpi, 300))
        except Exception as e:
            logger.warning("LaTeX render failed (%s); falling back to matplotlib if possible. Error: %s", type(e).__name__, e)
            # fall back to matplotlib below

    # Try matplotlib route
    try:
        return _matplotlib_render(expr, out_path, dpi=dpi, fontsize=fontsize)
    except Exception as e:
        # If matplotlib can't render and pdflatex is available, try latex route
        logger.warning("Matplotlib mathtext failed (%s). Trying full LaTeX if available. Error: %s", type(e).__name__, e)
        if shutil.which("pdflatex") and HAVE_PDF2IMAGE:
            return _latex_render(expr, out_path, dpi=max(dpi, 300))
        else:
            raise


# small demo: create a synthetic page by pasting multiple rendered expressions
def make_synthetic_page(out_dir, page_name="page_0001.png", n_eq=5, dpi=150):
    from PIL import Image, ImageDraw
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    page_width, page_height = 1240, 1754  # A4-ish at medium DPI
    bg = Image.new("RGB", (page_width, page_height), "white")
    for i in range(n_eq):
        if i % 2 == 0:
            expr = r"\nabla \cdot \mathbf{E} = \rho / \varepsilon_0"
            prefer_latex = False
        else:
            # correct matrix expression (single backslashes for LaTeX row separator)
            expr = r"\begin{pmatrix} a & b \\ c & d \end{pmatrix}"
            prefer_latex = True
        tmp = out_dir / f"tmp_eq_{i}.png"
        render_mathtext(expr, str(tmp), dpi=dpi, prefer_latex=prefer_latex)
        eq = Image.open(tmp)
        # random paste location
        import random
        maxx = page_width - eq.width - 50
        maxy = page_height - eq.height - 50
        x = random.randint(50, max(50, maxx))
        y = random.randint(50, max(50, maxy))
        bg.paste(eq, (x, y))
    out_path = out_dir / page_name
    bg.save(out_path)
    return out_path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="detector/data/images/synth", help="Output directory")
    p.add_argument("--n", default=20, type=int, help="Number of synthetic pages")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    for i in range(args.n):
        name = f"page_{i:04d}.png"
        make_synthetic_page(args.out_dir, page_name=name)
    print("Generated synthetic pages in", args.out_dir)
