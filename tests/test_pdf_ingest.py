import os
from pathlib import Path

import pytest

from equation_scribe.pdf_ingest import (
    load_pdf,
    page_image,
    page_layout,
    pdf_to_px_transform,
    find_equation_spans,
)
from equation_scribe.validate import validate_latex


# -----------------------------------------------------------------------------
# PDF sample configuration
# -----------------------------------------------------------------------------
# We look for a sample PDF in the repo by default, but allow overriding via
# the PDF_SAMPLE environment variable.
#
# This keeps tests fully portable and avoids any machine-specific paths.

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent

# Adjust this to whatever sample you want as the default
DEFAULT_PDF = PROJECT_ROOT / "data" / (
    "Research_on_SAR_Imaging_Simulation_Based_on_Time-Domain_"
    "Shooting_and_Bouncing_Ray_Algorithm.pdf"
)

PDF_SAMPLE = Path(os.getenv("PDF_SAMPLE", str(DEFAULT_PDF)))


@pytest.mark.skipif(
    not PDF_SAMPLE.exists(),
    reason=(
        "No sample PDF found. Set PDF_SAMPLE env var to a valid PDF path or "
        "place a test PDF under data/."
    ),
)
def test_load_and_render():
    doc = load_pdf(PDF_SAMPLE, dpi=200)
    assert doc.n_pages > 0

    # Render first page and check basic properties
    img = page_image(doc, 0, dpi=200)
    assert img is not None
    assert img.width > 0
    assert img.height > 0


@pytest.mark.skipif(
    not PDF_SAMPLE.exists(),
    reason=(
        "No sample PDF found. Set PDF_SAMPLE env var to a valid PDF path or "
        "place a test PDF under data/."
    ),
)
def test_pdf_to_px_transform_is_within_bounds():
    doc = load_pdf(PDF_SAMPLE, dpi=200)
    spans = page_layout(doc, 0)
    img = page_image(doc, 0, dpi=200)

    # Take a few spans and ensure their bbox maps into image coordinates
    for span in spans[:20]:
        (x0, y0, x1, y1) = span["bbox"]
        x0p, y0p, x1p, y1p = pdf_to_px_transform(img.size, (x0, y0, x1, y1))

        x_left, x_right = sorted((x0p, x1p))
        y_top, y_bottom = sorted((y0p, y1p))

        assert 0 <= x_left < x_right <= img.width
        assert 0 <= y_top < y_bottom <= img.height


@pytest.mark.skipif(
    not PDF_SAMPLE.exists(),
    reason=(
        "No sample PDF found. Set PDF_SAMPLE env var to a valid PDF path or "
        "place a test PDF under data/."
    ),
)
def test_equation_detection_and_validation():
    doc = load_pdf(PDF_SAMPLE, dpi=200)
    spans = page_layout(doc, 0)
    eqs = find_equation_spans(spans)

    # We don't assert a specific number of equations (PDF-dependent),
    # but we do sanity-check that the LaTeX validator returns something meaningful.
    for e in eqs:
        res = validate_latex(e["text"])
        assert res.ok or res.errors  # Either it parses or yields structured errors
