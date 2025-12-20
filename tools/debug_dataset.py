import argparse
import subprocess
import tempfile
import os
import random
import re
from pathlib import Path
from tqdm import tqdm

# --- 1. MINIMAL CLEANER ---
def clean_latex_formula(tex: str) -> str:
    if not tex: return ""
    return tex.strip()

# --- 2. MINIMAL TEMPLATE ---
TEX_TEMPLATE = r"""
\documentclass{article}
\usepackage[utf8]{inputenc}
% \usepackage{amsmath}
% \usepackage{amsthm}
% \usepackage{amssymb}
% \usepackage{amsfonts}
% \usepackage{bm}
% \usepackage{mathrsfs}
% \usepackage{color}

% %% COMPATIBILITY HACKS %%
% \makeatletter
% \renewcommand{\pmatrix}[1]{\left(\begin{matrix}#1\end{matrix}\right)}
% \renewcommand{\matrix}[1]{\begin{matrix}#1\end{matrix}}
% \renewcommand{\cases}[1]{\begin{cases}#1\end{cases}}
% \makeatother

\pagestyle{empty}
\begin{document}
__CONTENT__
\end{document}
"""

OUTER_ENVS = {
    "equation", "equation*", "align", "align*", "gather", "gather*", 
    "multline", "multline*", "eqnarray", "eqnarray*", "flalign", "flalign*"
}

def try_render(latex_str: str, debug_dir: Path, idx: int) -> str:
    # 1. SPLIT COMMENT FROM CODE
    # Find the first % that isn't escaped (isn't \%)
    match = re.search(r"(?<!\\)%", latex_str)
    
    if match:
        split_idx = match.start()
        math_part = latex_str[:split_idx]
        comment_part = latex_str[split_idx:]
    else:
        math_part = latex_str
        comment_part = ""

    stripped_math = math_part.strip()
    should_wrap = True
    
    # 2. CHECK OUTER ENVS (Only on the math part)
    # If the math part is empty (e.g. the line was just a comment), we shouldn't wrap
    if not stripped_math:
        should_wrap = False
    elif stripped_math.startswith(r"\begin"):
        m = re.match(r"\\begin\s*\{([^\}]+)\}", stripped_math)
        if m and m.group(1) in OUTER_ENVS:
            should_wrap = False
    
    # 3. APPLY WRAPPER (Before the comment!)
    if should_wrap and stripped_math:
        # We wrap the math part, then append the comment back outside the math mode
        content = f"\\[ {stripped_math} \\]" + comment_part
    else:
        content = stripped_math + comment_part

    # Create .tex file
    tex_path = debug_dir / f"debug_{idx}.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(TEX_TEMPLATE.replace("__CONTENT__", content))
        
    # Run pdflatex
    cmd = ["pdflatex", "-interaction=nonstopmode", f"-output-directory={debug_dir}", str(tex_path)]
    
    try:
        res = subprocess.run(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            timeout=5,
            check=False
        )
    except subprocess.TimeoutExpired:
        return "TIMEOUT"

    # Capture Log for Diagnostics
    log_content = ""
    log_path = debug_dir / f"debug_{idx}.log"
    if log_path.exists():
        with open(log_path, "r", encoding="latin-1", errors="ignore") as f:
            log_content = f.read()

    # --- 4. NULL OUTPUT CHECKS ---
    if "No pages of output" in log_content:
        return "NO OUTPUT (Empty?)"
    
    pdf_path = debug_dir / f"debug_{idx}.pdf"
    if not pdf_path.exists() or pdf_path.stat().st_size == 0:
        return "NO PDF GENERATED"

    if res.returncode == 0:
        return "OK"
    
    # Return first error from log
    for line in log_content.splitlines():
        if line.startswith("!"):
            return line
            
    return "UNKNOWN ERROR"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--formulas", required=True)
    parser.add_argument("--sample", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading {args.formulas}...")
    try:
        with open(args.formulas, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        with open(args.formulas, "r", encoding="latin-1") as f:
            lines = f.readlines()

    # Minimal Clean
    cleaned = []
    for line in lines:
        c = clean_latex_formula(line.strip())
        if len(c) > 5:
            cleaned.append(c)

    # Sample
    random.seed(args.seed)
    sample_indices = sorted(random.sample(range(len(cleaned)), min(args.sample, len(cleaned))))
    
    print(f"Testing {len(sample_indices)} formulas...")
    
    failures = []
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmp_path = Path(tmpdirname)
        for idx in tqdm(sample_indices):
            res = try_render(cleaned[idx], tmp_path, idx)
            if res != "OK":
                failures.append({"idx": idx, "latex": cleaned[idx], "error": res})

    print("\n" + "="*40)
    print(f"FAILURES: {len(failures)} / {len(sample_indices)}")
    print("="*40)
    
    if failures:
        print("\nTOP 10 FAILURES (Full Dump):")
        for f in failures[:10]:
            print(f"[{f['idx']}] ERROR: {f['error']}")
            print(f"LATEX: {f['latex']}")
            print("-" * 40)

if __name__ == "__main__":
    main()