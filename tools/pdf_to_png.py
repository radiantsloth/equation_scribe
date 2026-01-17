#!/usr/bin/env python3
"""
Small CLI tool: convert PDF pages -> one PNG file per page.

Usage:
    python tools/pdf_to_png.py input.pdf out_dir
    python tools/pdf_to_png.py input.pdf out_dir --dpi 300
    python tools/pdf_to_png.py input.pdf out_dir --pages 1-3,5 --dpi 150 --overwrite
    python tools/pdf_to_png.py input.pdf out_dir --zoom 1.5

Requirements (preferred):
    pip install pymupdf pillow

Fallback (if pymupdf not installed):
    pip install pdf2image pillow
    and install poppler (pdftoppm) on your system.
"""
from pathlib import Path
import argparse
import sys


def parse_pages_arg(pages_str):
    """Parse strings like '1-3,5,7-9' into a sorted list of 0-based page indices."""
    pages = set()
    for part in (pages_str or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            try:
                a, b = int(a), int(b)
            except ValueError:
                raise ValueError(f"Invalid page range token: {part}")
            if a < 1 or b < 1 or b < a:
                raise ValueError(f"Invalid page range: {part}")
            pages.update(range(a - 1, b))
        else:
            try:
                p = int(part)
            except ValueError:
                raise ValueError(f"Invalid page token: {part}")
            if p < 1:
                raise ValueError(f"Invalid page number: {part}")
            pages.add(p - 1)
    return sorted(pages)


def save_with_pymupdf(pdf_path: Path, out_dir: Path, pages, dpi: int = 150, zoom: float = None, fmt="png", overwrite=False):
    import fitz  # PyMuPDF
    doc = fitz.open(str(pdf_path))
    n = doc.page_count
    if pages is None:
        page_indices = range(0, n)
    else:
        page_indices = pages

    if zoom is not None:
        scale = float(zoom)
    else:
        # PDF points are 72 per inch
        scale = float(dpi) / 72.0

    for i in page_indices:
        if not (0 <= i < n):
            print(f"[skip] page {i+1} out of range (1-{n})", file=sys.stderr)
            continue
        page = doc.load_page(i)
        mat = fitz.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=mat, alpha=False)  # RGB
        out_path = out_dir / f"page_{i+1:04d}.{fmt}"
        if out_path.exists() and not overwrite:
            print(f"[exists] {out_path} (use --overwrite to replace)")
            continue
        # get PNG bytes and write
        data = pix.tobytes("png")
        out_path.write_bytes(data)
        print(f"[wrote] {out_path}")
    doc.close()


def save_with_pdf2image(pdf_path: Path, out_dir: Path, pages, dpi: int = 150, zoom: float = None, fmt="png", overwrite=False):
    # pdf2image always rasterizes by DPI; zoom is translated into DPI if provided as float (>0)
    from pdf2image import convert_from_path
    from PIL import Image
    if zoom is not None:
        # translate zoom to DPI (72 points per inch)
        dpi = int(round(float(zoom) * 72))

    if pages is None:
        imgs = convert_from_path(str(pdf_path), dpi=dpi)
        for idx, img in enumerate(imgs):
            out_path = out_dir / f"page_{idx+1:04d}.{fmt}"
            if out_path.exists() and not overwrite:
                print(f"[exists] {out_path} (use --overwrite to replace)")
                continue
            img.save(out_path, format=fmt.upper())
            print(f"[wrote] {out_path}")
    else:
        for i in pages:
            pnum = i + 1  # pdf2image is 1-based for pages
            try:
                imgs = convert_from_path(str(pdf_path), dpi=dpi, first_page=pnum, last_page=pnum)
            except Exception as e:
                print(f"[error] failed to render page {pnum}: {e}", file=sys.stderr)
                continue
            if not imgs:
                print(f"[warning] no image for page {pnum}", file=sys.stderr)
                continue
            img = imgs[0]
            out_path = out_dir / f"page_{pnum:04d}.{fmt}"
            if out_path.exists() and not overwrite:
                print(f"[exists] {out_path} (use --overwrite to replace)")
                continue
            img.save(out_path, format=fmt.upper())
            print(f"[wrote] {out_path}")


def main(argv=None):
    p = argparse.ArgumentParser(description="Convert PDF -> PNG (one file per page). Prefer PyMuPDF.")
    p.add_argument("pdf", help="Path to input PDF")
    p.add_argument("out_dir", help="Output directory for PNGs")
    p.add_argument("--dpi", type=int, default=150, help="Rasterization DPI (default: 150). Ignored if --zoom is given.")
    p.add_argument("--zoom", type=float, default=None, help="Zoom factor (overrides --dpi). E.g. 1.5")
    p.add_argument("--pages", type=str, default=None, help="Pages to convert, 1-based. e.g. '1-3,5,7'")
    p.add_argument("--fmt", type=str, default="png", help="Output format (default png)")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    args = p.parse_args(argv)

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}", file=sys.stderr)
        sys.exit(2)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pages = None
    if args.pages:
        try:
            pages = parse_pages_arg(args.pages)
        except ValueError as e:
            print(f"Error parsing --pages: {e}", file=sys.stderr)
            sys.exit(2)

    # try PyMuPDF first
    try:
        import fitz  # type: ignore
        use = "pymupdf"
    except Exception:
        use = None

    if use == "pymupdf":
        try:
            save_with_pymupdf(pdf_path, out_dir, pages, dpi=args.dpi, zoom=args.zoom, fmt=args.fmt, overwrite=args.overwrite)
            return
        except Exception as e:
            print(f"[warning] pymupdf rendering failed: {e}", file=sys.stderr)
            print("[info] attempting fallback to pdf2image...", file=sys.stderr)

    # fallback to pdf2image
    try:
        from pdf2image import convert_from_path  # type: ignore
        save_with_pdf2image(pdf_path, out_dir, pages, dpi=args.dpi, zoom=args.zoom, fmt=args.fmt, overwrite=args.overwrite)
        return
    except Exception as e:
        print(f"[error] pdf2image fallback failed: {e}", file=sys.stderr)

    print("No available renderer found. Install pymupdf (recommended) or pdf2image+poppler.", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
