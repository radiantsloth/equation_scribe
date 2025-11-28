#!/usr/bin/env python3
"""
detector/synthetic_coco.py

Generate synthetic pages with rendered LaTeX equations and produce COCO
annotations (instances_all.json). For each generated page we also write a
page-level .meta.json file containing the inserted equations (latex + bbox).

Usage:
  python detector/synthetic_coco.py \
    --out-images detector/data/images/synth \
    --out-anns detector/data/annotations/instances_all.json \
    --n-pages 100 --eqs-per-page 5 --dpi 150
"""
from pathlib import Path
import argparse
import json
import random
from PIL import Image

# import render_mathtext from render_latex (must be in detector/)
from detector.render_latex import render_mathtext

def bbox_to_coco(x0, y0, x1, y1):
    w = max(0.0, x1 - x0)
    h = max(0.0, y1 - y0)
    return [float(x0), float(y0), float(w), float(h)]

def make_synthetic_dataset(out_images_dir: Path, out_annotations: Path,
                           n_pages: int = 50, eqs_per_page: int = 5, dpi: int = 150):
    out_images_dir.mkdir(parents=True, exist_ok=True)
    annotations = []
    images = []
    categories = [{"id":1, "name":"display"}, {"id":2, "name":"inline"}]

    image_id = 1
    ann_id = 1

    pool = [
        r"E = mc^2",
        r"\nabla \cdot \mathbf{E} = \rho / \varepsilon_0",
        r"\frac{d}{dx}\sin(x) = \cos(x)",
        r"\int_{0}^{1} x^2 \, dx",
        r"\begin{pmatrix} a & b \\ c & d \end{pmatrix}",
        r"\sum_{n=0}^{\infty} \frac{x^n}{n!}",
        r"a^2 + b^2 = c^2",
        r"\alpha + \beta = \gamma",
        r"\sqrt{\frac{1}{2}}",
        r"\left( \frac{\partial^2}{\partial x^2} + \frac{\partial^2}{\partial y^2} \right) u = 0",
    ]

    random.seed(0)

    for pg in range(n_pages):
        page_name = f"page_{pg:04d}.png"
        page_path = out_images_dir / page_name
        page_w, page_h = 1240, 1754
        bg = Image.new("RGB", (page_w, page_h), "white")

        page_records = []
        for i in range(eqs_per_page):
            expr = random.choice(pool)
            tmp_png = out_images_dir / f"tmp_{pg}_{i}.png"
            prefer_latex = "\\begin" in expr or "\n" in expr or "\\\\[" in expr
            try:
                render_mathtext(expr, str(tmp_png), dpi=dpi, prefer_latex=prefer_latex)
            except Exception:
                # fallback to matplotlib rendering if pdflatex or pdf2image fails
                try:
                    render_mathtext(expr, str(tmp_png), dpi=dpi, prefer_latex=False)
                except Exception as e:
                    print("Skipping expr; render failed:", expr, e)
                    continue
            eq_img = Image.open(tmp_png)
            ew, eh = eq_img.size
            maxx = max(60, page_w - ew - 60)
            maxy = max(60, page_h - eh - 60)
            x = random.randint(60, maxx)
            y = random.randint(60, maxy)
            bg.paste(eq_img, (x, y))

            x0, y0, x1, y1 = float(x), float(y), float(x + ew), float(y + eh)
            page_records.append({"latex": expr, "bbox": [x0, y0, x1, y1], "type": "display"})
            try:
                tmp_png.unlink()
            except Exception:
                pass

        # Save page and page-level metadata file
        bg.save(page_path)
        meta = {"file_name": str(page_path), "width": page_w, "height": page_h, "eqs": page_records}
        meta_path = page_path.with_suffix(".meta.json")
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        # Append image record and annotations to COCO lists
        images.append({"id": image_id, "file_name": str(page_path), "width": page_w, "height": page_h})
        for rec in page_records:
            x0, y0, x1, y1 = rec["bbox"]
            coco_bbox = bbox_to_coco(x0, y0, x1, y1)
            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": 1 if rec.get("type") == "display" else 2,
                "bbox": coco_bbox,
                "area": coco_bbox[2] * coco_bbox[3],
                "iscrowd": 0,
                "segmentation": []
            })
            ann_id += 1
        image_id += 1

    coco = {"info": {"description": "Synthetic equation dataset"},
            "licenses": [],
            "images": images,
            "annotations": annotations,
            "categories": categories}
    out_annotations.parent.mkdir(parents=True, exist_ok=True)
    with open(out_annotations, "w", encoding="utf-8") as fh:
        json.dump(coco, fh, indent=2)
    print(f"Wrote {len(images)} images and {len(annotations)} annotations to {out_annotations}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-images", default="detector/data/images/synth", help="Output images folder")
    p.add_argument("--out-anns", default="detector/data/annotations/instances_all.json", help="Output COCO annotations JSON")
    p.add_argument("--n-pages", type=int, default=50)
    p.add_argument("--eqs-per-page", type=int, default=5)
    p.add_argument("--dpi", type=int, default=150)
    args = p.parse_args()
    make_synthetic_dataset(Path(args.out_images), Path(args.out_anns),
                           n_pages=args.n_pages, eqs_per_page=args.eqs_per_page, dpi=args.dpi)

if __name__ == "__main__":
    main()
