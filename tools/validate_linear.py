import argparse
import subprocess
import tempfile
import sys
import re
from pathlib import Path
from tqdm import tqdm


# --- 1. ROBUST CLEANER ---
def clean_latex_formula(tex: str) -> str:
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
    # We loop until all \label{...} occurrences are removed
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
            # CHANGE: Replace with a space " " to prevent token merging
            # e.g. \protect\label{x}u -> \protect u (Valid) instead of \protectu (Invalid)
            tex = tex[:start_idx] + " " + tex[end_idx+1:]
        else:
            # Malformed label (no closing brace found). 
            # Stop to prevent infinite loop, or just strip the rest.
            # For safety, we just stop modifying.
            break

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
    # Note: latex_str is already cleaned of comments/labels by the loader
    stripped = latex_str.strip()
    should_wrap = True
    
    # Check Outer Envs
    if not stripped:
        should_wrap = False
    elif stripped.startswith(r"\begin"):
        m = re.match(r"\\begin\s*\{([^\}]+)\}", stripped)
        if m and m.group(1) in OUTER_ENVS:
            should_wrap = False
    
    # Apply Wrapper
    if should_wrap and stripped:
        content = f"\\[ {stripped} \\]"
    else:
        content = stripped

    # Create .tex file
    tex_path = debug_dir / f"validate_{idx}.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(TEX_TEMPLATE.replace("__CONTENT__", content))
        
    # Run pdflatex
    cmd = ["pdflatex", "-interaction=nonstopmode", f"-output-directory={debug_dir}", str(tex_path)]
    
    try:
        res = subprocess.run(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            timeout=20,
            check=False
        )
    except subprocess.TimeoutExpired:
        return "TIMEOUT"

    # Diagnostics
    log_content = ""
    log_path = debug_dir / f"validate_{idx}.log"
    if log_path.exists():
        with open(log_path, "r", encoding="latin-1", errors="ignore") as f:
            log_content = f.read()

    # Fail if no pages (empty output)
    if "No pages of output" in log_content:
        return "NO OUTPUT (Empty?)"
    
    # Fail if no PDF generated
    pdf_path = debug_dir / f"validate_{idx}.pdf"
    if not pdf_path.exists() or pdf_path.stat().st_size == 0:
        if "Fatal error" in log_content or "!" in log_content:
             for line in log_content.splitlines():
                if line.startswith("!"):
                    return line
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
    parser.add_argument("--start", type=int, default=0, help="Start list index")
    args = parser.parse_args()

    print(f"Loading {args.formulas}...")
    try:
        with open(args.formulas, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        with open(args.formulas, "r", encoding="latin-1") as f:
            lines = f.readlines()

    # Clean and Filter (Linear scan)
    all_formulas = []
    original_indices = []
    
    print("Cleaning formulas...")
    for i, line in enumerate(lines):
        c = clean_latex_formula(line.strip())
        # ONLY add if content remains. This handles the "won't count as error" requirement.
        if len(c) > 0: 
            all_formulas.append(c)
            original_indices.append(i)

    print(f"Kept {len(all_formulas)} formulas (out of {len(lines)} lines).")
    print(f"Validating linearly starting from index {args.start}...")
    
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmp_path = Path(tmpdirname)
        
        # Iterate linearly
        for i in range(args.start, len(all_formulas)):
            latex = all_formulas[i]
            original_idx = original_indices[i]
            
            # Print progress every 100
            if i % 100 == 0:
                print(f"Processing index {i}/{len(all_formulas)}...", end="\r")

            res = try_render(latex, tmp_path, original_idx)
            
            if res != "OK":
                print("\n" + "!"*40)
                print(f"FAILURE AT LIST INDEX {i} (File Line {original_idx})")
                print(f"ERROR: {res}")
                print("-" * 20)
                print(f"LATEX (Cleaned):\n{latex}")
                print("!"*40)
                sys.exit(1) # STOP IMMEDIATELY

    print("\nAll formulas passed validation!")

if __name__ == "__main__":
    main()