# equation_scribe/autodetect_equations.py
"""Heuristic equation autodetection with shared core persistence helpers.

Record schema written to equations.jsonl (one JSON object per line):
{
    "uid": "<unique-id>",
    "paper_id": "<paper-id>",
    "page_index": <int>,
    "bbox_pdf": [x0, y0, x1, y1],  # PDF point coords
    "bbox_px": [x0, y0, x1, y1],   # pixel coords of rendered page (if available)
    "latex": "<latex string>" or null,
    "confidence": <float>,
    "symbols": { ... }  # optional glossary mapping
}
"""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time
import shutil
from typing import List, Dict, Any

from .pdf_ingest import load_pdf, page_size_points, page_layout_with_ocr
from .detect import find_equation_candidates
from .store import canonical_hash

try:
    from equation_scribe_core.io import register_paper, write_jsonl
except ModuleNotFoundError:
    core_src = Path(__file__).resolve().parents[1] / "packages" / "core" / "src"
    if str(core_src) not in sys.path:
        sys.path.insert(0, str(core_src))
    from equation_scribe_core.io import register_paper, write_jsonl

# NOTE: we intentionally do not save per-record inside the loop.
# Instead we collect all_records and write the JSONL once at the end,
# so we can protect existing files and register atomically.

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
        A list of equation records (as plain dicts) that were detected.
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

            # Collect record; write will be handled after detection completes.
            all_records.append(record)

    return all_records


def write_detected_equations(
    records: List[Dict[str, Any]],
    *,
    pdf_path: str | Path,
    paper_id: str,
    data_root: str | Path,
    force: bool = False,
) -> Path | None:
    """Write autodetected records and update the shared paper index.

    The current CLI behavior is preserved:
    - do not overwrite an existing profile unless ``force`` is set
    - when ``force`` is set, rotate the old ``equations.jsonl`` into the paper
      directory with a timestamped ``.bak`` suffix before writing the new file
    - always update the shared index after a successful write
    """

    profiles_root = Path(data_root)
    paper_dir = profiles_root / paper_id
    paper_dir.mkdir(parents=True, exist_ok=True)
    out_path = paper_dir / "equations.jsonl"

    if out_path.exists() and not force:
        print(f"ERROR: {out_path} already exists. Use --force to overwrite.")
        return None

    if out_path.exists() and force:
        ts = int(time.time())
        bak = paper_dir / f"equations.jsonl.bak.{ts}"
        shutil.move(str(out_path), str(bak))
        print(f"Backed up existing {out_path} to {bak}")

    write_jsonl(out_path, records)
    print(f"Wrote {len(records)} records to {out_path}")

    pdf_basename = Path(pdf_path).name
    try:
        register_paper(
            profiles_root,
            paper_id=paper_id,
            pdf_basename=pdf_basename,
            profiles_dir=paper_id,
            num_equations=len(records),
            force=force,
        )
        print(f"Updated index.json under {profiles_root} for paper_id={paper_id!r}")
    except RuntimeError as e:
        print(f"WARNING: index not updated: {e}")

    return out_path


if __name__ == "__main__":
    # Simple CLI wrapper so you can run this from the command line.
    import argparse

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
    ap.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing equations.jsonl and index entry if present",
    )

    args = ap.parse_args()
    cfg = AutoDetectConfig(min_score=args.min_score)

    # Run detection (collect records)
    print(f"Running autodetect on {args.pdf} ...")
    records = autodetect_equations(args.pdf, args.paper_id, args.data_root, cfg=cfg)
    print(json.dumps({"detected": len(records)}, indent=2))

    write_detected_equations(
        records,
        pdf_path=args.pdf,
        paper_id=args.paper_id,
        data_root=args.data_root,
        force=args.force,
    )
