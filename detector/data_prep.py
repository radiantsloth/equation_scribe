# detector/data_prep.py
import json
from pathlib import Path
from PIL import Image
import argparse
from collections import defaultdict
import time

def bbox_to_coco(x0, y0, x1, y1):
    w = max(0.0, x1 - x0)
    h = max(0.0, y1 - y0)
    return [x0, y0, w, h]

def build_coco(images_info, annotations, categories):
    return {
        "info": {"description": "Equation detector dataset", "version": "1.0", "year": time.localtime().tm_year},
        "licenses": [],
        "images": images_info,
        "annotations": annotations,
        "categories": categories
    }

def convert_from_profiles(
    profiles_jsonl_path,
    pdf_path=None,
    page_images_dir=None,
    out_annotations_path="detector/data/annotations/instances.json",
    categories=None
):
    """
    Convert equations.jsonl (profiles) to a simplified COCO annotations file.
    - profiles_jsonl_path: a file or folder with equations.jsonl files (or single file)
    - pdf_path: if provided, used to compute pdf->pixel transform (requires equation_scribe.pdf_ingest)
    - page_images_dir: folder with page images named like page_{page:04d}.png
    """
    profiles_jsonl_path = Path(profiles_jsonl_path)
    page_images_dir = Path(page_images_dir) if page_images_dir else None
    annotations = []
    images_info = []
    categories = categories or [{"id":1,"name":"display"}, {"id":2, "name":"inline"}]
    cat_name_to_id = {c["name"]: c["id"] for c in categories}

    # Attempt to import pdf helpers if available
    try:
        from equation_scribe.pdf_ingest import load_pdf, pdf_to_px_transform
        have_pdf_helpers = True
    except Exception:
        have_pdf_helpers = False

    image_id_map = {}   # (pdfname,page) -> image_id
    next_image_id = 1
    next_ann_id = 1

    def add_image_record(fname, width, height):
        nonlocal next_image_id
        img_record = {"id": next_image_id, "file_name": str(fname), "width": width, "height": height}
        images_info.append(img_record)
        image_id_map[str(fname)] = next_image_id
        next_image_id += 1
        return img_record["id"]

    # If profiles_jsonl_path is a folder, find equations.jsonl files inside
    files = []
    if profiles_jsonl_path.is_dir():
        files = list(profiles_jsonl_path.rglob("equations.jsonl"))
    elif profiles_jsonl_path.is_file():
        files = [profiles_jsonl_path]
    else:
        raise RuntimeError("profiles_jsonl_path not found")

    for f in files:
        with f.open("r", encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                paper_id = rec.get("paper_id")
                boxes = rec.get("boxes", [])
                # If PDF path provided, try to load and map
                doc = None
                if pdf_path:
                    try:
                        pdf_path = Path(pdf_path)
                        doc = load_pdf(pdf_path)
                    except Exception:
                        doc = None
                for b in boxes:
                    page = b.get("page", 0)
                    bbox_pdf = b.get("bbox_pdf")  # expected [x0,y0,x1,y1] in PDF pts
                    cls = b.get("cls", "display")
                    # Map PDF bbox to pixels
                    if doc and have_pdf_helpers and bbox_pdf:
                        pdf2px, px2pdf = pdf_to_px_transform(doc, page)
                        x0p, y0p = pdf2px(bbox_pdf[0], bbox_pdf[1])
                        x1p, y1p = pdf2px(bbox_pdf[2], bbox_pdf[3])
                    else:
                        # Fallback: assume bbox_pdf are pixel coords
                        x0p, y0p, x1p, y1p = bbox_pdf

                    # Determine image filename (try page_images_dir)
                    if page_images_dir:
                        img_path = Path(page_images_dir) / f"page_{page:04d}.png"
                    else:
                        # fallback: name by paper and page
                        img_path = Path(f"{paper_id}_page_{page:04d}.png")

                    # create image record if not exists
                    if str(img_path) not in image_id_map:
                        if img_path.exists():
                            with Image.open(img_path) as im:
                                w,h = im.size
                        else:
                            # fallback sizes if image not available
                            w,h = int(x1p)+10, int(y1p)+10
                        add_image_record(str(img_path), w, h)

                    img_id = image_id_map[str(img_path)]
                    coco_bbox = bbox_to_coco(float(x0p), float(y0p), float(x1p), float(y1p))
                    # category id mapping - use class name if present
                    if isinstance(cls, str):
                        cat_id = cat_name_to_id.get(cls, 1)
                    else:
                        cat_id = int(cls)+1

                    ann = {
                        "id": next_ann_id,
                        "image_id": img_id,
                        "category_id": cat_id,
                        "bbox": coco_bbox,
                        "area": coco_bbox[2]*coco_bbox[3],
                        "iscrowd": 0,
                        "segmentation": []
                    }
                    annotations.append(ann)
                    next_ann_id += 1

    coco = build_coco(images_info, annotations, categories)
    out_path = Path(out_annotations_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(coco, fh, indent=2)
    print("Wrote COCO annotations to", out_path)
