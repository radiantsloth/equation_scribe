import subprocess
import os
import re
from pathlib import Path

# The problematic string (hardcoded exactly as you pasted it)
FORMULA = r"\left<E\right>_{ren}^{mode}=\int_0^{\infty}d\omega  \frac1{2}\omega \left[N(\omega)-N_0(\omega)\right],\label{modesum}"

# The Template (Standard/Robust version)
TEX_TEMPLATE = r"""
\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage{amsmath}
\usepackage{amsfonts}
\usepackage{amssymb}
\pagestyle{empty}
\begin{document}
%s
\end{document}
"""

OUTER_ENVS = {
    "equation", "equation*", "align", "align*", "gather", "gather*", 
    "multline", "multline*", "eqnarray", "eqnarray*", "flalign", "flalign*"
}

def test_render():
    print(f"Testing Formula:\n{FORMULA}\n")

    # 1. Simulate the Wrapper Logic
    stripped = FORMULA.strip()
    should_wrap = True
    
    if stripped.startswith(r"\begin"):
        m = re.match(r"\\begin\s*\{([^\}]+)\}", stripped)
        if m and m.group(1) in OUTER_ENVS:
            should_wrap = False
            print("Detected Outer Environment -> NO WRAP")
        else:
            print(f"Detected Inner Environment ({m.group(1)}) -> WRAPPING in \[ \]")
    
    if should_wrap:
        content = f"\\[ {FORMULA} \\]"
    else:
        content = FORMULA

    # 2. Construct LaTeX
    tex_content = TEX_TEMPLATE % content
    
    # 3. Write to disk
    debug_dir = Path("debug_single_output")
    debug_dir.mkdir(exist_ok=True)
    tex_path = debug_dir / "test.tex"
    
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex_content)
        
    print(f"\nWrote LaTeX to {tex_path}")
    print("-" * 20)
    print(tex_content)
    print("-" * 20)

    # 4. Run PDFLaTeX
    cmd = ["pdflatex", "-interaction=nonstopmode", f"-output-directory={debug_dir}", str(tex_path)]
    
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
        
        if res.returncode == 0:
            print("\n✅ SUCCESS! The formula rendered correctly.")
            print(f"Check {debug_dir / 'test.pdf'}")
        else:
            print("\n❌ FAILURE! pdflatex returned non-zero code.")
            # Print the log's error message
            log_path = debug_dir / "test.log"
            if log_path.exists():
                with open(log_path, "r", encoding="latin-1") as f:
                    for line in f:
                        if line.strip().startswith("!"):
                            print(f"LaTeX Error: {line.strip()}")
                            break
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")

if __name__ == "__main__":
    test_render()