import os
import sys
from pathlib import Path

# Ensure we can import from the current directory
# (This allows running from repo root via: python equation_scribe/detector/render_samples.py)
sys.path.append(os.getcwd())

try:
    from equation_scribe.detector.render_latex import render_mathtext
except ImportError:
    # Fallback if running directly inside detector/ directory
    sys.path.append(str(Path(__file__).parent))
    from equation_scribe.detector.render_latex import render_mathtext

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

def main():
    # Output directory
    out_dir = Path("detector/data/samples")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Rendering {len(SAMPLE_EQUATIONS)} samples to {out_dir}...")

    for i, eq in enumerate(SAMPLE_EQUATIONS):
        filename = f"sample_{i:02d}.png"
        out_path = out_dir / filename
        
        print(f"[{i}] Rendering: {eq}")
        
        try:
            # DPI 300 for crisp text
            render_mathtext(eq, out_path, dpi=300, fontsize=28)
            print(f"    -> Saved {out_path}")
        except Exception as e:
            print(f"    !! Failed to render: {e}")

    print("\nDone. Check folder:", out_dir.resolve())

if __name__ == "__main__":
    main()