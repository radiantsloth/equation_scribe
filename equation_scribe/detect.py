# equation_scribe/detect.py
from __future__ import annotations
import re
from typing import List, Dict, Any, Tuple

# ... (Keep your existing MATH_GLYPHS, GREEK, LATEX_HINTS, OP_CHARS constants here) ...
MATH_GLYPHS = set("∑∫∂∇±≈≠≤≥∞√→←×•°≃≅≡⊂⊃⊆⊇∈∉∪∩∧∨¬⇒⇔⊗⊕…")
GREEK = set("αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ")
LATEX_HINTS = ("\\frac", "\\cdot", "\\nabla", "\\sum", "\\int", "\\partial", "\\sqrt", "\\leq", "\\geq")
OP_CHARS = set("=+-/*^_|()[]{}<>")

def _mathy_score(s: str) -> float:
    # (Keep your existing _mathy_score function)
    s = s or ""
    n = len(s)
    if n == 0: return 0.0
    m = sum(ch in MATH_GLYPHS or ch in GREEK or ch in OP_CHARS for ch in s)
    m += sum(h in s for h in LATEX_HINTS) * 3
    alpha = sum(ch.isalpha() for ch in s)
    return (m + 1) / (alpha + 5)

def cluster_spans_into_lines(spans: List[Dict[str, Any]], y_tolerance: float = 3.0, x_gap_threshold: float = 40.0) -> List[List[Dict[str, Any]]]:
    """
    Groups words into lines, respecting column gaps.
    1. Sort by Y.
    2. Group words that are vertically close.
    3. Within those vertical groups, sort by X.
    4. Split if there is a horizontal gap > x_gap_threshold (e.g. the gutter in IEEE papers).
    """
    if not spans:
        return []

    # 1. Sort by top Y
    sorted_spans = sorted(spans, key=lambda b: b["bbox_pdf"][1])
    
    lines = []
    current_line = [sorted_spans[0]]
    
    # 2. Vertical Clustering
    for span in sorted_spans[1:]:
        prev = current_line[-1]
        # Check vertical overlap/proximity
        if abs(span["bbox_pdf"][1] - prev["bbox_pdf"][1]) < y_tolerance:
            current_line.append(span)
        else:
            lines.append(current_line)
            current_line = [span]
    lines.append(current_line)

    # 3. Horizontal Splitting (Column Detection)
    final_segments = []
    for line in lines:
        # Sort by Left X
        line.sort(key=lambda b: b["bbox_pdf"][0])
        
        current_segment = [line[0]]
        for span in line[1:]:
            prev_span = current_segment[-1]
            gap = span["bbox_pdf"][0] - prev_span["bbox_pdf"][2] # x0_curr - x1_prev
            
            if gap > x_gap_threshold:
                # Gutter detected! Start new segment.
                final_segments.append(current_segment)
                current_segment = [span]
            else:
                current_segment.append(span)
        final_segments.append(current_segment)

    return final_segments

def find_equation_candidates(spans: List[Dict[str, Any]], page_width: float) -> List[Dict[str, Any]]:
    """
    New logic: Cluster by line AND column, then score.
    """
    if not spans: return []

    # Use the new column-aware clusterer
    segments = cluster_spans_into_lines(spans)
    
    candidates = []
    for seg in segments:
        # Compute Union BBox
        xs0 = [w["bbox_pdf"][0] for w in seg]
        ys0 = [w["bbox_pdf"][1] for w in seg]
        xs1 = [w["bbox_pdf"][2] for w in seg]
        ys1 = [w["bbox_pdf"][3] for w in seg]
        x0, y0, x1, y1 = min(xs0), min(ys0), max(xs1), max(ys1)
        
        text = " ".join(w["text"] for w in seg)
        score = _mathy_score(text)

        # Heuristic: IEEE equations are often indented or centered *within their column*
        # But for now, raw mathy score is a good enough filter
        if score >= 0.3:
            candidates.append({"text": text, "bbox_pdf": (x0, y0, x1, y1), "score": round(score, 3)})

    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates
