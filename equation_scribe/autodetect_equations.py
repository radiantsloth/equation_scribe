# equation_scribe/autodetect_equations.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

from .pdf_ingest import load_pdf, page_size_points, page_layout_with_ocr
from .detect import find_equation_candidates
from .store import save_equation, canonical_hash


@dataclass
class AutoDetectConfig:
    """
    Configuration for the heuristic auto-detector.

    Attributes:
        min_score: Minimum candidate score (from detect.find_equation_candidates)
                   to keep as a likely equation.
    """
    min_score: float = 0.6


def autodetect_equations(
    pdf_path: str | Path,
    paper_id: str,
    data_root: str | Path,
    cfg: AutoDetectConfig | None = None,
) -> List[Dict[str, Any]]:
    """
    Run a first-pass, heuristic equation detector over a PDF.

    Args:
        pdf_path: Path to the input PDF file.
        paper_id: Identifier for this paper (used as directory name under data_root).
        data_root: Root directory that holds per-paper subdirectories.
                   For your web app, this should match `equation_scribe_web/paper_profiles`.
        cfg: Optional AutoDetectConfig to tweak thresholds.

    Returns:
        A list of equation records (as plain dicts) that were detected and saved.
        Each record has the shape expected by the web backend's EquationRecord:
          {
            "eq_uid": str,
            "paper_id": str,
            "latex": str,
            "notes": str,
            "boxes": [
              { "page": int, "bbox_pdf": [x0, y0, x1, y1] }
            ]
          }
    """
    cfg = cfg or AutoDetectConfig()
    pdf_path = Path(pdf_path)
    data_root = Path(data_root)

    # Load the PDF (this verifies that it exists and has pages)
    doc = load_pdf(pdf_path)
    all_records: List[Dict[str, Any]] = []

    for page_index in range(doc.num_pages):
        # Unified span extraction: text-layer first, OCR fallback otherwise
        spans = page_layout_with_ocr(doc, page_index)
        if not spans:
            # Nothing to work with on this page
            continue

        # Page width in PDF points, used for center-ness heuristic in detect.py
        page_width, _ = page_size_points(doc, page_index)

        # Use your existing "mathy" detector
        candidates = find_equation_candidates(spans, page_width)

        for cand in candidates:
            # Some earlier code already filters by score, but we can enforce it here too.
            if cand.get("score", 0.0) < cfg.min_score:
                continue

            text = cand.get("text", "")
            x0, y0, x1, y1 = cand["bbox_pdf"]

            # Generate a stable ID for the equation based on its text.
            eq_uid = canonical_hash(text)

            record: Dict[str, Any] = {
                "eq_uid": eq_uid,
                "paper_id": paper_id,
                # For now, use the raw extracted text as a placeholder for LaTeX.
                # Later, we'll run a LaTeX conversion / SymPy validation pass.
                "latex": text,
                "notes": "",
                "boxes": [
                    {
                        "page": page_index,
                        "bbox_pdf": [float(x0), float(y0), float(x1), float(y1)],
                    }
                ],
            }

            # Save to JSONL via the shared store module.
            # This will write to: data_root / paper_id / "equations.jsonl"
            save_equation(data_root, paper_id, record)
            all_records.append(record)

    return all_records


if __name__ == "__main__":
    # Simple CLI wrapper so you can run this from the command line.
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Heuristic equation auto-detector")
    ap.add_argument("--pdf", required=True, help="Path to input PDF")
    ap.add_argument("--paper-id", required=True, help="Identifier for this paper")
    ap.add_argument(
        "--data-root",
        required=True,
        help=(
            "Directory that contains per-paper subdirs "
            "(e.g. C:/Data/repos/equation_scribe_web/equation_scribe_web/paper_profiles)"
        ),
    )
    ap.add_argument(
        "--min-score",
        type=float,
        default=0.6,
        help="Minimum candidate score to keep (higher = stricter)",
    )

    args = ap.parse_args()
    cfg = AutoDetectConfig(min_score=args.min_score)

    records = autodetect_equations(args.pdf, args.paper_id, args.data_root, cfg)
    print(json.dumps({"detected": len(records)}, indent=2))
