#!/usr/bin/env python3
"""
render_latex.py — render LaTeX math to PNG.

Approach:
  1. Prefer pdflatex + standalone package for complex LaTeX environments (matrix,
     arrays, multi-line constructs). Use pdf2image/poppler to rasterize.
  2. Fall back to matplotlib.mathtext rendering for simpler expressions.

Requirements:
  - pdflatex (MiKTeX or TeX Live) for full LaTeX rendering
  - poppler (pdftoppm) and pdf2image (optional) to convert PDF to PNG
  - matplotlib for fallback rendering

API:
  - render_mathtext(expr, out_path, dpi=150, prefer_latex=True) -> None
"""

import os
import shutil
import subprocess
import tempfile
import re
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

# --- 1. ROBUST PREAMBLE ---
# Includes packages and hacks for im2latex compatibility
TEX_TEMPLATE = r"""
\documentclass{article}
\usepackage[utf8]{inputenc}
# \usepackage{amsmath}
# \usepackage{amsthm}
# \usepackage{amssymb}
# \usepackage{amsfonts}
# \usepackage{bm}
# \usepackage{mathrsfs}
# \usepackage{color}

# %% COMPATIBILITY HACKS for im2latex %%
# \makeatletter
# \renewcommand{\pmatrix}[1]{\left(\begin{matrix}#1\end{matrix}\right)}
# \renewcommand{\matrix}[1]{\begin{matrix}#1\end{matrix}}
# % \cases usually works, but this ensures it maps to standard amsmath
# \renewcommand{\cases}[1]{\begin{cases}#1\end{cases}}
# \makeatother

\pagestyle{empty}
\begin{document}
%s
\end{document}
"""

# Environments that already provide math mode (don't wrap these in \[ \])
OUTER_ENVS = {
    "equation", "equation*", "align", "align*", "gather", "gather*", 
    "multline", "multline*", "eqnarray", "eqnarray*", "flalign", "flalign*"
}

def _cleanup(work_dir, job_name):
    """Removes temporary LaTeX artifacts."""
    for ext in [".aux", ".log", ".tex", ".pdf"]:
        f = work_dir / f"{job_name}{ext}"
        if f.exists():
            try:
                os.unlink(f)
            except OSError:
                pass

def _latex_render(latex: str, out_path: str, dpi: int = 150):
    """
    Renders LaTeX to an image file using pdflatex and pdf2image.
    """
    if not HAVE_PDF2IMAGE:
        raise RuntimeError("pdf2image module not found. Please install it to render LaTeX.")

    # --- 2. SMART WRAPPING LOGIC ---
    stripped = latex.strip()
    should_wrap = True
    
    # Check if the formula starts with a known Outer Environment
    if stripped.startswith(r"\begin"):
        m = re.match(r"\\begin\s*\{([^\}]+)\}", stripped)
        if m and m.group(1) in OUTER_ENVS:
            should_wrap = False
    
    if should_wrap:
        content = f"\\[ {latex} \\]"
    else:
        content = latex

    # Prepare file paths
    out_path = Path(out_path).resolve()
    work_dir = out_path.parent
    job_name = out_path.stem
    tex_file = work_dir / f"{job_name}.tex"
    pdf_file = work_dir / f"{job_name}.pdf"
    
    # Write .tex file
    with open(tex_file, "w", encoding="utf-8") as f:
        f.write(TEX_TEMPLATE % content)
    
    # Run pdflatex
    # -interaction=nonstopmode ensures it doesn't pause for user input on error
    cmd = ["pdflatex", "-interaction=nonstopmode", f"-output-directory={work_dir}", str(tex_file)]
    
    try:
        # --- 3. TIMEOUT ADDED (10s) ---
        subprocess.run(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            check=True, 
            timeout=10
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        _cleanup(work_dir, job_name)
        raise RuntimeError(f"LaTeX render failed or timed out: {e}")

    # Convert generated PDF to PNG
    try:
        images = convert_from_path(str(pdf_file), dpi=dpi)
        if images:
            # Save the first page (formulas are usually single page)
            images[0].save(out_path)
        else:
            raise RuntimeError("PDF generated but contained no pages.")
    except Exception as e:
        _cleanup(work_dir, job_name)
        raise RuntimeError(f"Image conversion failed: {e}")

    # Clean up temp files
    _cleanup(work_dir, job_name)
    
    return out_path  # <--- RETURN ADDED


def _matplotlib_render(expr: str, out_path: str, dpi: int = 150, fontsize: int = 10):
    """Fallback renderer: uses matplotlib mathtext to render an expression."""
    # Wrap expression in $...$ if not already math mode (matplotlib expects math mode).
    tex = expr
    if not (tex.startswith("$") and tex.endswith("$")):
        if not (tex.startswith(r"\(") or tex.startswith(r"\[")):
            tex = f"${tex}$"

    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0.0, 0.0, tex, fontsize=fontsize)
    # Tight bbox to crop around the rendered equation
    try:
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    finally:
        plt.close(fig)


def render_mathtext(expr: str, out_path: str, dpi: int = 150, prefer_latex: bool = True):
    """
    Main entry point. 
    Attempts to render using LaTeX (pdflatex) if prefer_latex is True.
    Falls back to Matplotlib if LaTeX fails or is not preferred.
    """
    # 1. Try LaTeX (Priority)
    if prefer_latex and HAVE_PDF2IMAGE:
        try:
            _latex_render(expr, out_path, dpi=dpi)
            return
        except Exception as e:
            # If LaTeX fails, just fall through to Matplotlib
            # print(f"LaTeX failed for {expr[:20]}...: {e}")
            pass

    # 2. Fallback to Matplotlib
    try:
        # Use fontsize 10 to match IEEE body text
        _matplotlib_render(expr, out_path, dpi=dpi, fontsize=10)
    except Exception as e:
        # If both fail, raise error so the caller knows to skip this formula
        raise RuntimeError(f"Both renderers failed for: {expr[:30]}...")