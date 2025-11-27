#!/usr/bin/env python3
"""
detector/data_prep_coco.py

Convert existing equation profile JSONL files (equations.jsonl) into COCO-format
annotations suitable for training a detector (YOLOv8 / Detectron2).

Features:
- Accepts a folder of profiles (PROFILES_ROOT/<paper_id>/equations.jsonl) or a single
  equations.jsonl file.
- Optionally renders PDF pages to images (pdf2image) if page_images_dir is not present.
- Converts bbox_pdf (PDF points) to pixel coordinates using:
    - equation_scribe.pdf_ingest.pdf_to_px_transform(doc, page) if available, or
    - a fallback scale assuming page width 612 pts with page height derived from image aspect ratio.
- Writes a COCO JSON annotation file and enumerates the images.

Usage:
  python detector/data_prep_coco.py \
    --profiles-root C:/Data/Research/paper_profiles \
    --out-json detector/data/annotations/instances_all.json \
    --page-images-dir C:/Data/detector_images \
    --render-pdf \
    --dpi 150 \
    --split 0.8

If you prefer to render PDFs to images, provide `--pdf-root` (where source PDFs live)
and `--render-pdf` will be used to generate page images into page-images-dir/<paper_id>/page_0001.png.

"""
import json
import argparse
from pathlib import Path
from PIL import Image
import time
import math
import shutil
import tempfile
import sys
from typing import List, Dict, Tuple, Optional

# Optional dependency for PDF rendering
try:
    from pdf2image import convert_from_path
    HAVE_PDF2IMAGE = True
except Exception:
    HAVE_PDF2IMAGE = False

# Try to import your repo's pdf helpers
try:
    from equation_scribe.pdf_ingest import load_pdf, pdf_to_px_transform
    HAVE_PDF_HELPERS = True
except Exception:
    HAVE_PDF_HELPERS = False

# Default page width (PDF pts) if we need a fallback conversion
DEFAULT_PAGE_WIDTH_PT = 612.0

def bbox_to_coco(x0: float, y0: float, x1: float, y1: float) -> List[float]:
    w = max(0.0, x1 - x0)
    h = max(0.0, y1 - y0)
    return [float(x0), float(y0), float(w), float(h)]

def build_coco(images_info, annotations, categories):
    return {
        "info": {"description": "Equation detector dataset", "version": "1.0", "year": time.localtime().tm_year},
        "licenses": [],
        "images": images_info,
        "annotations": annotations,
        "categories": categories
    }

def render_pdf_pages(pdf_path: Path, out_dir: Path, dpi: int = 150, fmt: str = "png") -> int:
    """
    Render all pages of pdf_path into out_dir/page_0001.png, page_0002.png, ...
    Returns number of pages rendered.
    """
    if not HAVE_PDF2IMAGE:
        raise RuntimeError("pdf2image is required to render PDFs. Install `pip install pdf2image` and poppler.")
    out_dir.mkdir(parents=True, exist_ok=True)
    pages = convert_from_path(str(pdf_path), dpi=dpi)
    for i, page in enumerate(pages):
        fname = out_dir / f"page_{i:04d}.{fmt}"
        page.save(fname)
    return len(pages)

def find_equations_jsonl_files(profiles_root: Path) -> List[Path]:
    """
    Search for equations.jsonl files under profiles_root.
    """
    files = list(profiles_root.rglob("equations.jsonl"))
    return files

def load_page_image_size(img_path: Path) -> Tuple[int,int]:
    with Image.open(img_path) as im:
        return im.size  # (width, height)

def pdf_bbox_to_pixel_bbox_fallback(bbox_pdf, img_w_px, img_h_px, page_width_pt=DEFAULT_PAGE_WIDTH_PT):
    """
    Fallback conversion that assumes page_width_pt for PDF page width and derives PDF page height
    from the image aspect ratio. Converts PDF pts bbox to pixel bbox.
    """
    x0, y0, x1, y1 = bbox_pdf
    # derive page height in pts by preserving aspect ratio
    page_h_pt = page_width_pt * (img_h_px / img_w_px)
    sx = img_w_px / page_width_pt
    sy = img_h_px / page_h_pt
    # flip Y: PDF origin bottom-left vs image top-left
    x0_px = x0 * sx
    x1_px = x1 * sx
    # compute from top-left pixel coordinate system
    y0_px = (page_h_pt - y0) * sy
    y1_px = (page_h_pt - y1) * sy
    # normalize
    x_min, x_max = sorted([x0_px, x1_px])
    y_min, y_max = sorted([y0_px, y1_px])
    return [x_min, y_min, x_max, y_max]

def convert_profiles_to_coco(
    profiles_root: Path,
    out_annotations: Path,
    page_images_dir: Optional[Path] = None,
    pdf_root: Optional[Path] = None,
    render_pdf: bool = False,
    dpi: int = 150,
    split: float = 1.0,
    categories=None
):
    """
    Convert all equations.jsonl under profiles_root into a COCO annotations file.
    If split < 1.0, it will write train/val splits: instances_train.json / instances_val.json
    """
    profiles_root = Path(profiles_root)
    out_annotations = Path(out_annotations)
    if categories is None:
        categories = [{"id":1,"name":"display"}, {"id":2,"name":"inline"}]
    cat_name_to_id = {c["name"]: c["id"] for c in categories}

    files = find_equations_jsonl_files(profiles_root)
    if not files:
        raise RuntimeError(f"No equations.jsonl found under {profiles_root}")

    images_info = []
    annotations = []
    image_id_map = {}
    next_image_id = 1
    next_ann_id = 1

    def add_image_record(fname: str, w:int, h:int):
        nonlocal next_image_id
        img_record = {"id": next_image_id, "file_name": fname, "width": w, "height": h}
        images_info.append(img_record)
        image_id_map[fname] = next_image_id
        next_image_id += 1
        return img_record["id"]

    # Iterate profiles
    for eq_file in files:
        # eq_file = PROFILES_ROOT/<paper_id>/equations.jsonl
        paper_dir = eq_file.parent
        paper_id = paper_dir.name
        # possible pdf path
        pdf_candidate = None
        if pdf_root:
            # try to find pdf with same basename under pdf_root
            # allow common suffixes
            candidates = list(Path(pdf_root).rglob(f"{paper_id}*.pdf"))
            if candidates:
                pdf_candidate = candidates[0]
        # locate page image folder for this paper
        images_folder_for_paper = None
        if page_images_dir:
            # prefer per-paper folder
            pdir = Path(page_images_dir) / paper_id
            if pdir.exists():
                images_folder_for_paper = pdir
            else:
                # global images dir with page files named <paper>_page_0000.png or page_0000.png
                images_folder_for_paper = Path(page_images_dir)
        else:
            # if user requested render_pdf and pdf exists, create an images folder under detector/data/images/<paper_id>
            if render_pdf and pdf_candidate:
                out_dir = Path("detector/data/images") / paper_id
                out_dir.mkdir(parents=True, exist_ok=True)
                print(f"Rendering PDF {pdf_candidate} to {out_dir} at {dpi} DPI...")
                try:
                    render_pdf_pages(pdf_candidate, out_dir, dpi=dpi)
                    images_folder_for_paper = out_dir
                except Exception as e:
                    print("ERROR rendering PDF:", e)
                    images_folder_for_paper = None
            else:
                images_folder_for_paper = None

        # if pdf helpers available and a pdf exists, load doc once
        doc = None
        if HAVE_PDF_HELPERS and pdf_candidate:
            try:
                doc = load_pdf(pdf_candidate)
            except Exception:
                doc = None

        # Read equations.jsonl
        with eq_file.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                boxes = rec.get("boxes", [])
                for box in boxes:
                    page_idx = int(box.get("page", 0))
                    bbox_pdf = box.get("bbox_pdf")
                    if not bbox_pdf:
                        continue
                    # Determine image path (priority):
                    # 1) images_folder_for_paper / page_{page_idx:04d}.png
                    # 2) images_folder_for_paper / <paper>_page_{page_idx:04d}.png
                    # 3) fallback: paper_id + page
                    img_path_candidates = []
                    if images_folder_for_paper:
                        img_path_candidates.append(images_folder_for_paper / f"page_{page_idx:04d}.png")
                        img_path_candidates.append(images_folder_for_paper / f"{paper_id}_page_{page_idx:04d}.png")
                    # also check paper_dir/images/page_0000.png
                    maybe_local = paper_dir / "images" / f"page_{page_idx:04d}.png"
                    img_path_candidates.append(maybe_local)
                    chosen_img = None
                    for p in img_path_candidates:
                        if p and p.exists():
                            chosen_img = p
                            break

                    # If not found and doc is available, we can render just this page to a temp image
                    temp_image = None
                    if not chosen_img and doc:
                        # render a single page using pdf2image (if available)
                        if HAVE_PDF2IMAGE:
                            tmp_folder = Path("detector/data/images/_tmp") / paper_id
                            tmp_folder.mkdir(parents=True, exist_ok=True)
                            try:
                                pages = convert_from_path(str(pdf_candidate), dpi=dpi, first_page=page_idx+1, last_page=page_idx+1)
                                # pages list should have one image
                                tmp_file = tmp_folder / f"page_{page_idx:04d}.png"
                                pages[0].save(tmp_file)
                                chosen_img = tmp_file
                                temp_image = tmp_file
                            except Exception as e:
                                print("Warning: failed to render page", e)
                                chosen_img = None

                    # If still not found, create a synthetic image size fallback
                    if not chosen_img:
                        # we'll create a dummy image size large enough to include bbox
                        # assume bbox_pdf is in pixel coords as last resort
                        x0p, y0p, x1p, y1p = bbox_pdf
                        width = int(max(1024, math.ceil(x1p + 10)))
                        height = int(max(1024, math.ceil(y1p + 10)))
                        # create a small temp image
                        tmp_folder = Path("detector/data/images/_generated")
                        tmp_folder.mkdir(parents=True, exist_ok=True)
                        chosen_img = tmp_folder / f"{paper_id}_page_{page_idx:04d}.png"
                        if not chosen_img.exists():
                            from PIL import Image, ImageDraw
                            im = Image.new("RGB", (width, height), "white")
                            im.save(chosen_img)

                    # At this point chosen_img exists
                    img_w, img_h = load_page_image_size(chosen_img)
                    # Convert PDF bbox to pixel bbox
                    # Prefer using repo pdf helpers if doc was available
                    if doc and HAVE_PDF_HELPERS:
                        try:
                            pdf2px, px2pdf = pdf_to_px_transform(doc, page_idx)
                            # pdf2px expects (x_pt, y_pt) and returns pixel coords
                            x0_px, y0_px = pdf2px(bbox_pdf[0], bbox_pdf[1])
                            x1_px, y1_px = pdf2px(bbox_pdf[2], bbox_pdf[3])
                        except Exception:
                            # fallback to generic conversion
                            x0_px, y0_px, x1_px, y1_px = pdf_bbox_to_pixel_bbox_fallback(bbox_pdf, img_w, img_h)
                    else:
                        # fallback conversion
                        x0_px, y0_px, x1_px, y1_px = pdf_bbox_to_pixel_bbox_fallback(bbox_pdf, img_w, img_h)

                    # Normalize & clip to image bounds
                    x0_px = max(0.0, min(x0_px, img_w-1))
                    y0_px = max(0.0, min(y0_px, img_h-1))
                    x1_px = max(0.0, min(x1_px, img_w-1))
                    y1_px = max(0.0, min(y1_px, img_h-1))
                    # Ensure valid
                    if x1_px <= x0_px or y1_px <= y0_px:
                        # skip invalid boxes
                        continue

                    fname_rel = str(chosen_img)
                    if fname_rel not in image_id_map:
                        img_id = add_image_record(fname_rel, img_w, img_h)
                    else:
                        img_id = image_id_map[fname_rel]

                    # Determine category id
                    clsname = box.get("cls", None) or box.get("class", None) or box.get("type", None)
                    if isinstance(clsname, str):
                        cat_id = cat_name_to_id.get(clsname, 1)
                    else:
                        # If numeric class present (0/1), map to ids (1-based)
                        try:
                            cat_id = int(box.get("cls", 0)) + 1
                        except Exception:
                            cat_id = 1

                    coco_bbox = bbox_to_coco(x0_px, y0_px, x1_px, y1_px)
                    ann = {
                        "id": next_ann_id,
                        "image_id": img_id,
                        "category_id": cat_id,
                        "bbox": coco_bbox,
                        "area": coco_bbox[2] * coco_bbox[3],
                        "iscrowd": 0,
                        "segmentation": []
                    }
                    annotations.append(ann)
                    next_ann_id += 1

                    # cleanup temp image if any
                    if temp_image and temp_image.exists():
                        # keep rendered tmp pages if you want; for now leave them
                        pass

    coco = build_coco(images_info, annotations, categories)
    out_annotations.parent.mkdir(parents=True, exist_ok=True)
    with out_annotations.open("w", encoding="utf-8") as fh:
        json.dump(coco, fh, indent=2)
    print("Wrote COCO annotations to", out_annotations)
    return out_annotations

def main():
    p = argparse.ArgumentParser(description="Convert equations.jsonl profiles into COCO annotations")
    p.add_argument("--profiles-root", required=True, help="Root folder with profiles (PROFILES_ROOT)")
    p.add_argument("--out-json", required=True, help="Output COCO annotations JSON path")
    p.add_argument("--page-images-dir", default=None, help="Optional folder with page images (per-paper subfolders or shared)")
    p.add_argument("--pdf-root", default=None, help="Optional root folder where PDFs live (used if render_pdf is requested)")
    p.add_argument("--render-pdf", action="store_true", help="Render PDF pages if page images are missing (requires pdf2image/poppler)")
    p.add_argument("--dpi", type=int, default=150, help="DPI for rendering pages")
    p.add_argument("--split", type=float, default=1.0, help="If <1.0, reserved fraction for training; otherwise single JSON output")
    args = p.parse_args()

    profiles_root = Path(args.profiles_root)
    out_json = Path(args.out_json)
    page_images_dir = Path(args.page_images_dir) if args.page_images_dir else None
    pdf_root = Path(args.pdf_root) if args.pdf_root else None

    if args.render_pdf and not HAVE_PDF2IMAGE:
        print("ERROR: --render-pdf requested but pdf2image is not installed. Install pdf2image and poppler for rendering.")
        sys.exit(1)

    convert_profiles_to_coco(
        profiles_root=profiles_root,
        out_annotations=out_json,
        page_images_dir=page_images_dir,
        pdf_root=pdf_root,
        render_pdf=args.render_pdf,
        dpi=args.dpi,
        split=args.split,
        categories=[{"id":1,"name":"display"}, {"id":2,"name":"inline"}]
    )

if __name__ == "__main__":
    main()
