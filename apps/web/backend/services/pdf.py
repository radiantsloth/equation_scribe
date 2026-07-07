from pathlib import Path
from typing import Tuple, Dict, Any
import fitz  # PyMuPDF

def page_count(pdf_path: Path) -> int:
    with fitz.open(pdf_path) as doc:
        return doc.page_count

def page_size_points(pdf_path: Path, page_index: int) -> Tuple[float, float]:
    with fitz.open(pdf_path) as doc:
        page = doc.load_page(page_index)
        r = page.rect
        return float(r.width), float(r.height)

def render_page_png(pdf_path: Path, page_index: int, zoom: float = 1.5) -> bytes:
    with fitz.open(pdf_path) as doc:
        page = doc.load_page(page_index)
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        return pix.tobytes("png")

def page_meta(pdf_path: Path, page_index: int, zoom: float = 1.5) -> Dict[str, Any]:
    w_pt, h_pt = page_size_points(pdf_path, page_index)
    with fitz.open(pdf_path) as doc:
        page = doc.load_page(page_index)
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        w_px, h_px = pix.width, pix.height
    return {
        "width_pts": w_pt,
        "height_pts": h_pt,
        "width_px": w_px,
        "height_px": h_px,
        "zoom": zoom,
    }
