#!/usr/bin/env python3
"""
render_latex.py — render LaTeX math to PNG.
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
# CHANGE: Use Figure directly to avoid global state memory leaks
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
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
TEX_TEMPLATE = r"""
\documentclass[12pt]{article}
\usepackage[utf8]{inputenc}
\usepackage{amsmath}
\usepackage{amsthm}
\usepackage{amssymb}
\usepackage{amsfonts}
\usepackage{bm}
\usepackage{mathrsfs}
\usepackage{color}

%% COMPATIBILITY HACKS for im2latex %%
\makeatletter
\renewcommand{\pmatrix}[1]{\left(\begin{matrix}#1\end{matrix}\right)}
\renewcommand{\matrix}[1]{\begin{matrix}#1\end{matrix}}
\renewcommand{\cases}[1]{\begin{cases}#1\end{cases}}
\makeatother

\pagestyle{empty}
\begin{document}
%s
\end{document}
"""

OUTER_ENVS = {
    "equation", "equation*", "align", "align*", "gather", "gather*", 
    "multline", "multline*", "eqnarray", "eqnarray*", "flalign", "flalign*"
}

def _latex_render(latex, out_path, dpi=150):
    """
    Renders LaTeX to an image using pdflatex + pdftoppm.
    """
    if not HAVE_PDF2IMAGE:
        raise RuntimeError("pdf2image module not found.")

    stripped = latex.strip()
    should_wrap = True
    
    # Smart Wrapping
    if stripped.startswith(r"\begin"):
        import re
        m = re.match(r"\\begin\s*\{([^\}]+)\}", stripped)
        if m and m.group(1) in OUTER_ENVS:
            should_wrap = False
    
    if should_wrap:
        content = f"\\[ {latex} \\]"
    else:
        content = latex

    # Prepare .tex file
    tex_source = TEX_TEMPLATE % content
    out_path = Path(out_path).resolve()
    work_dir = out_path.parent
    job_name = out_path.stem
    tex_file = work_dir / f"{job_name}.tex"
    
    with open(tex_file, "w", encoding="utf-8") as f:
        f.write(tex_source)
    
    # Run pdflatex with INCREASED TIMEOUT
    cmd = ["pdflatex", "-interaction=nonstopmode", f"-output-directory={work_dir}", str(tex_file)]
    
    try:
        subprocess.run(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            check=True, 
            timeout=30
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        # Capture the log for debugging
        log_file = work_dir / f"{job_name}.log"
        error_details = str(e)
        if log_file.exists():
            with open(log_file, "r", encoding="latin-1", errors="ignore") as lf:
                log_content = lf.read()
                # extract lines starting with !
                errors = [line for line in log_content.splitlines() if line.startswith("!")]
                if errors:
                    error_details += f"\nLaTeX Error Log: {errors[:3]}"

        # Cleanup and raise
        if tex_file.exists(): os.unlink(tex_file)
        raise RuntimeError(f"LaTeX render failed: {error_details}")

    pdf_file = work_dir / f"{job_name}.pdf"
    if not pdf_file.exists():
        raise RuntimeError("pdflatex did not produce PDF output.")

    # convert to PNG using pdf2image
    try:
        pages = convert_from_path(str(pdf_file), dpi=dpi, fmt="png")
        if len(pages) == 0:
            raise RuntimeError("pdf2image did not return any pages")
        pages[0].convert("RGB").save(out_path)
    finally:
        # Cleanup artifacts
        for ext in [".aux", ".log", ".tex", ".pdf"]:
            f = work_dir / f"{job_name}{ext}"
            if f.exists():
                try: os.unlink(f)
                except: pass
                
    return out_path

def _matplotlib_render(expr: str, out_path: str, dpi: int = 150, fontsize: int = 20):
    """
    Fallback renderer: uses matplotlib mathtext.
    Uses Object-Oriented API (Figure) to avoid global state memory leaks.
    """
    tex = expr
    if not (tex.startswith("$") and tex.endswith("$")):
        tex = f"${tex}$"

    # CHANGE: Use Figure() directly instead of plt.figure()
    # This prevents the "More than 20 figures have been opened" warning
    # because these figures are never registered with the pyplot state machine.
    fig = Figure(figsize=(0.01, 0.01))
    FigureCanvasAgg(fig) # Attach a canvas so we can save
    
    fig.text(0.0, 0.0, tex, fontsize=fontsize)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    # No plt.close(fig) needed because we never opened it in pyplot!

def render_mathtext(expr, out_path, dpi=150, prefer_latex=False):
    # 1. Try LaTeX
    if prefer_latex and HAVE_PDF2IMAGE:
        try:
            _latex_render(expr, out_path, dpi=dpi)
            return
        except Exception as e:
            # print(f"pdflatex failed: {e}")
            pass

    # 2. Fallback to Matplotlib
    try:
        _matplotlib_render(expr, out_path, dpi=dpi)
    except Exception as e:
        # If both fail, raise a clear error
        raise RuntimeError(f"Both renderers failed for: {expr[:30]}...")