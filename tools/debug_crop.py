"""
Debug tool to verify PDF cropping coordinates.
Usage:
    python tools/debug_crop.py --paper_id "Research_on_SAR..." --page 0 --bbox "50,100,200,150"

This will output 'debug_crop_output.png' containing the cropped area.
"""
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from equation_scribe.pdf_ingest import load_pdf, page_image, pdf_to_px_transform, page_size_points
from backend.main import PAPERS_ROOT

def debug_crop_box(paper_id: str, page_idx: int, bbox_str: str):
    # 1. Resolve Path
    pdf_path = PAPERS_ROOT / f"{paper_id}.pdf"
    if not pdf_path.exists():
        print(f"Error: PDF not found at {pdf_path}")
        return

    print(f"Loading {pdf_path}...")
    doc = load_pdf(pdf_path)
    
    # 2. Parse bbox
    try:
        x0, y0, x1, y1 = map(float, bbox_str.split(","))
    except ValueError:
        print("Error: bbox must be x0,y0,x1,y1")
        return

    # 3. Setup transforms
    dpi = 150
    print(f"Generating page image at {dpi} DPI...")
    full_img = page_image(doc, page_idx, dpi=dpi)
    pdf2px, _ = pdf_to_px_transform(doc, page_idx, dpi=dpi)

    # 4. Transform Coordinates
    px0, py0 = pdf2px(x0, y0)
    px1, py1 = pdf2px(x1, y1)
    
    print(f"PDF Coords: ({x0}, {y0}) to ({x1}, {y1})")
    print(f"Px  Coords: ({px0}, {py0}) to ({px1}, {py1})")
    
    # 5. Crop
    # Ensure ordered coordinates for PIL
    crop_box = (min(px0, px1), min(py0, py1), max(px0, px1), max(py0, py1))
    crop_img = full_page_img.crop(crop_box)
    
    # 6. Save
    out_name = "debug_crop_output.png"
    crop_img.save(out_name)
    print(f"Saved crop to {out_name}. Check this image to verify alignment.")

    # Also save full page with a red rectangle drawn on it for context
    try:
        from PIL import ImageDraw
        draw = ImageDraw.Draw(full_img)
        draw.rectangle(crop_box, outline="red", width=3)
        full_img.save("debug_full_page_context.png")
        print("Saved debug_full_page_context.png with red box.")
    except ImportError:
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper_id", required=True, help="ID of the paper (filename without .pdf)")
    parser.add_argument("--page", type=int, default=0)
    parser.add_argument("--bbox", required=True, help="x0,y0,x1,y1")
    args = parser.parse_args()
    
    debug_crop_box(args.paper_id, args.page, args.bbox)