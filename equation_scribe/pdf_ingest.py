
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import re
import fitz  # PyMuPDF
import pdfplumber
from PIL import Image
# Optional dependency for OCR-based layout on scanned PDFs
try:
    import pytesseract
    _HAS_TESSERACT = True
except Exception:
    pytesseract = None
    _HAS_TESSERACT = False

PT_PER_INCH = 72.0  # PDF coordinate system: 72 points per inch


@dataclass
class PdfDoc:
    """
    PdfDoc: small data container for a PDF file.

    Attributes:
        path (Path): Filesystem path to the PDF.
        num_pages (int): Number of pages in the PDF.
        dpi (int): Default DPI for rasterization.
    """
    path: Path
    num_pages: int
    dpi: int = 300

    @property
    def n_pages(self) -> int:
        """Compatibility alias expected by tests (doc.n_pages)."""
        return self.num_pages



def load_pdf(path: str | Path, dpi: int = 300) -> PdfDoc:
    """
    Open a PDF, verify it exists and has pages, return a PdfDoc.
    Args:
        path: File path to the PDF.
        dpi: Default DPI for downstream rendering.
    Returns:
        PdfDoc
    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the PDF has zero pages.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"PDF not found: {p}")
    with pdfplumber.open(str(p)) as pdf:
        n = len(pdf.pages)
    if n == 0:
        raise ValueError("PDF has zero pages")
    return PdfDoc(path=p, num_pages=n, dpi=dpi)


def page_size_points(doc: PdfDoc, i: int) -> Tuple[float, float]:
    """
    Return (width, height) of page i in PDF points.
    """
    if not (0 <= i < doc.num_pages):
        raise IndexError("page index out of range")
    with fitz.open(str(doc.path)) as pdf:
        page = pdf[i]
        rect = page.rect
        return float(rect.width), float(rect.height)


def page_image(doc: PdfDoc, i: int, dpi: Optional[int] = None) -> Image.Image:
    """
    Rasterize page i to an RGB Pillow Image at the given DPI.
    Args:
        doc: PdfDoc
        i: page index (0-based)
        dpi: Optional DPI override (defaults to doc.dpi)
    Returns:
        PIL.Image.Image
    """
    if not (0 <= i < doc.num_pages):
        raise IndexError("page index out of range")
    dpi = dpi or doc.dpi
    scale = dpi / PT_PER_INCH
    with fitz.open(str(doc.path)) as pdf:
        page = pdf[i]
        mat = fitz.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=mat, alpha=False)  # RGB
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return img


def pdf_to_px_transform(arg1, arg2=None, dpi: Optional[int] = None):
    """
    Return transformation functions for mapping between PDF point coordinates
    and pixel coordinates for a given PDF page.

    **CRITICAL COORDINATE SYSTEM NOTE:**
    This project standardizes on **Top-Left Origin** for both PDF points and Pixels.
    - PDF (pdfplumber): (0,0) is Top-Left. Y increases downwards.
    - Image (PIL/Fitz): (0,0) is Top-Left. Y increases downwards.
    
    This function generates a simple scaler without axis flipping.

    Args:
        arg1: Either a PdfDoc instance OR a tuple of (img_width_px, img_height_px)
        arg2: Either a page_index (int) OR a bbox tuple (x0, y0, x1, y1)
        dpi: Optional DPI override (defaults to doc.dpi)

    Returns:
        tuple: (pdf2px, px2pdf)
        - pdf2px(x_pt, y_pt) -> (x_px, y_px)
        - px2pdf(x_px, y_px) -> (x_pt, y_pt)
    """
    # Mode 2: img_size + bbox -> pixel bbox (Legacy/Test support)
    if isinstance(arg1, (tuple, list)) and arg2 is not None:
        img_w_px, img_h_px = int(arg1[0]), int(arg1[1])
        x0, y0, x1, y1 = arg2
        
        # Calculate scale based on standard width (assuming standard letter width 612pt)
        # In a real scenario, this is brittle without the actual PDF page size, 
        # but kept for compatibility with existing tests if they use this mode.
        PAGE_WIDTH_PT = 612.0
        sx = img_w_px / PAGE_WIDTH_PT
        sy = sx # Assume square aspect ratio scaling if we don't have page height

        # Simple scaling (No Flip)
        x0p = int(round(x0 * sx))
        x1p = int(round(x1 * sx))
        y0p = int(round(y0 * sy)) 
        y1p = int(round(y1 * sy))
        
        return (min(x0p, x1p), min(y0p, y1p), max(x0p, x1p), max(y0p, y1p))

    # Mode 1: doc (PdfDoc), page index (Standard Usage)
    doc = arg1
    page_index = arg2
    dpi = dpi or getattr(doc, "dpi", 300)
    
    # We actually don't need w_pt/h_pt for simple scaling, 
    # but we fetch them to ensure page_index is valid.
    w_pt, h_pt = page_size_points(doc, page_index)
    
    sx = dpi / PT_PER_INCH
    sy = dpi / PT_PER_INCH

    def pdf_to_px(x_pt: float, y_pt: float) -> Tuple[int, int]:
        """Maps PDF points (Top-Left) to Image Pixels (Top-Left)."""
        x_px = int(round(x_pt * sx))
        y_px = int(round(y_pt * sy))  # Removed (h_pt - y_pt) flip
        return x_px, y_px

    def px_to_pdf(x_px: int, y_px: int) -> Tuple[float, float]:
        """Maps Image Pixels (Top-Left) to PDF points (Top-Left)."""
        x_pt = x_px / sx
        y_pt = y_px / sy              # Removed flip logic
        return x_pt, y_pt

    return pdf_to_px, px_to_pdf



def page_layout(doc: PdfDoc, i: int) -> List[Dict[str, Any]]:
    """
    Extract word-level spans for page i using pdfplumber.

    Returns:
        List of dicts: { 'text': str, 'bbox_pdf': (x0,y0,x1,y1), 'page_index': int }
        Coordinates are in PDF points.
        For scanned/image-only PDFs, this will likely be an empty list.
    """
    if not (0 <= i < doc.num_pages):
        raise IndexError("page index out of range")
    spans: List[Dict[str, Any]] = []
    with pdfplumber.open(str(doc.path)) as pdf:
        page = pdf.pages[i]
        try:
            words = page.extract_words()
        except Exception:
            words = []
        for w in words:
            bbox_pdf = (float(w["x0"]), float(w["top"]), float(w["x1"]), float(w["bottom"]))
            spans.append({
                "text": w.get("text", ""),
                "bbox_pdf": bbox_pdf,
                "bbox": bbox_pdf,             # compatibility for tests that expect "bbox"
                "page_index": i,
            })
    return spans

def page_layout_ocr(doc: PdfDoc, i: int) -> List[Dict[str, Any]]:
    """
    OCR-based fallback for scanned / image-only PDFs.

    Strategy:
      * Render the page to an RGB image.
      * Run Tesseract to get word boxes in pixel coordinates.
      * Convert those into PDF point coordinates using px_to_pdf.
      * Return spans in the same format as page_layout().

    Returns:
        List of dicts:
          { 'text': str, 'bbox_pdf': (x0, y0, x1, y1), 'page_index': int }
    """
    if not _HAS_TESSERACT:
        # Tesseract not available; nothing we can do here.
        return []

    from pytesseract import Output

    if not (0 <= i < doc.num_pages):
        raise IndexError("page index out of range")

    # Render image at the document DPI
    img = page_image(doc, i, dpi=doc.dpi)

    # Coordinate transforms between PDF points and pixels
    pdf_to_px, px_to_pdf = pdf_to_px_transform(doc, i, dpi=doc.dpi)

    data = pytesseract.image_to_data(img, output_type=Output.DICT)
    n = len(data.get("text", []))
    spans: List[Dict[str, Any]] = []

    for idx in range(n):
        txt = (data["text"][idx] or "").strip()
        if not txt:
            continue

        # Confidence: filter out garbage OCR
        try:
            conf = float(data.get("conf", ["0"])[idx])
        except Exception:
            conf = 0.0
        if conf < 40:  # heuristic threshold; you can tune this
            continue

        # Tesseract gives pixel coords in the image's coordinate system:
        # left, top, width, height; origin is top-left of the image.
        x = int(data["left"][idx])
        y = int(data["top"][idx])
        w = int(data["width"][idx])
        h = int(data["height"][idx])

        x0_px, y0_px = x, y
        x1_px, y1_px = x + w, y + h

        # Map pixel coordinates back to PDF points
        x0_pt, y0_pt = px_to_pdf(x0_px, y0_px)
        x1_pt, y1_pt = px_to_pdf(x1_px, y1_px)

        # Normalize so that x0 < x1 and y0 < y1
        x0_pdf = float(min(x0_pt, x1_pt))
        x1_pdf = float(max(x0_pt, x1_pt))
        y0_pdf = float(min(y0_pt, y1_pt))
        y1_pdf = float(max(y0_pt, y1_pt))

        spans.append({
            "text": txt,
            "bbox_pdf": (x0_pdf, y0_pdf, x1_pdf, y1_pdf),
            "bbox": (x0_pdf, y0_pdf, x1_pdf, y1_pdf),
            "page_index": i,
        })

    return spans


def page_layout_with_ocr(doc: PdfDoc, i: int) -> List[Dict[str, Any]]:
    """
    High-level API: get spans for page i.

    * First, try to use the native text layer via page_layout().
    * If that returns no spans (common for scanned PDFs),
      fall back to OCR via page_layout_ocr().

    This function is what your auto-detector should call.
    """
    spans = page_layout(doc, i)
    if spans:
        return spans

    # Fallback for scanned pages
    ocr_spans = page_layout_ocr(doc, i)
    return ocr_spans


def find_equation_spans(spans):
    """
    Heuristic filter for spans that look like equations.
    Detects math delimiters ($...$, \(..\), \[..\]).
    """
    eq_spans = []
    math_pattern = re.compile(r"(\$.*?\$|\\\(.*?\\\)|\\\[.*?\\\])")
    for s in spans:
        if math_pattern.search(s["text"]):
            eq_spans.append(s)
    return eq_spans
