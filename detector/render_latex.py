# detector/render_latex.py
import random
import os
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

BASE_SNIPPETS = [
    r"E = mc^2",
    r"\nabla \cdot \mathbf{E} = \rho / \varepsilon_0",
    r"\frac{d}{dx}\sin(x) = \cos(x)",
    r"\int_{0}^{1} x^2 dx",
    r"A = \begin{pmatrix} a & b \\ c & d \end{pmatrix}",
    r"\sum_{n=0}^{\infty} \frac{x^n}{n!}"
]

def render_mathtext(expr, out_path, dpi=200, fontsize=28):
    """Render a single math expression to an image file using matplotlib."""
    fig = plt.figure(figsize=(3,1))
    plt.text(0.5, 0.5, f"${expr}$", fontsize=fontsize, ha='center', va='center')
    plt.axis('off')
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    # convert to RGB with pillow for consistency
    Image.open(out_path).convert("RGB").save(out_path)

def make_synthetic_page(out_dir, page_name="page_0001.png", n_eq=5, dpi=150):
    """Create a synthetic page by pasting a few rendered equations onto a white background."""
    from PIL import Image, ImageDraw
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    page_width, page_height = 1240, 1754  # A4-ish at medium DPI
    bg = Image.new("RGB", (page_width, page_height), "white")
    for i in range(n_eq):
        expr = random.choice(BASE_SNIPPETS)
        tmp = out_dir / f"tmp_eq_{i}.png"
        render_mathtext(expr, str(tmp), dpi=dpi)
        eq = Image.open(tmp)
        # paste randomly
        maxx = page_width - eq.width - 50
        maxy = page_height - eq.height - 50
        if maxx < 50 or maxy < 50:
            x, y = 50, 50
        else:
            x = random.randint(50, maxx)
            y = random.randint(50, maxy)
        bg.paste(eq, (x, y))
    out_path = out_dir / page_name
    bg.save(out_path)
    return out_path

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="detector/data/images/synth", help="Output directory")
    p.add_argument("--n", default=20, type=int, help="Number of synthetic pages")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    for i in range(args.n):
        name = f"page_{i:04d}.png"
        make_synthetic_page(args.out_dir, page_name=name)
    print("Generated synthetic pages in", args.out_dir)
