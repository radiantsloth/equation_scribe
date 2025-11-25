# equation_scribe/detect.py
from __future__ import annotations
import re
from typing import List, Dict, Any

MATH_GLYPHS = set("∑∫∂∇±≈≠≤≥∞√→←×•°≃≅≡⊂⊃⊆⊇∈∉∪∩∧∨¬⇒⇔⊗⊕…")
GREEK = set("αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ")
LATEX_HINTS = ("\\frac", "\\cdot", "\\nabla", "\\sum", "\\int", "\\partial", "\\sqrt", "\\leq", "\\geq")

OP_CHARS = set("=+-/*^_|()[]{}<>")

def _mathy_score(s: str) -> float:
    s = s or ""
    n = len(s)
    if n == 0: return 0.0
    m = sum(ch in MATH_GLYPHS or ch in GREEK or ch in OP_CHARS for ch in s)
    m += sum(h in s for h in LATEX_HINTS) * 3
    alpha = sum(ch.isalpha() for ch in s)
    return (m + 1) / (alpha + 5)

def find_equation_candidates(spans: List[Dict[str, Any]], page_width: float) -> List[Dict[str, Any]]:
    """
    Group words into rough 'lines' using their vertical positions, score for 'mathiness',
    filter to centered-ish lines (display equations), and return candidate bboxes.
    Each result: {'text': str, 'bbox_pdf': (x0,y0,x1,y1), 'score': float}
    """
    if not spans: return []

    # crude line clustering by y (pdfplumber "top" coordinate)
    by_y = {}
    BIN = 3.0
    for w in spans:
        y = float(w["bbox_pdf"][1])  # top
        key = round(y / BIN) * BIN
        by_y.setdefault(key, []).append(w)

    candidates = []
    for y_key, ws in by_y.items():
        # union bbox
        xs0 = [w["bbox_pdf"][0] for w in ws]
        ys0 = [w["bbox_pdf"][1] for w in ws]
        xs1 = [w["bbox_pdf"][2] for w in ws]
        ys1 = [w["bbox_pdf"][3] for w in ws]
        x0, y0, x1, y1 = min(xs0), min(ys0), max(xs1), max(ys1)
        text = " ".join(w["text"] for w in sorted(ws, key=lambda q: q["bbox_pdf"][0]))
        score = _mathy_score(text)

        # center-ness: distance of bbox center to page center (0..1)
        cx = 0.5 * (x0 + x1)
        center_dev = abs(cx - page_width / 2) / (page_width / 2)
        centered_bonus = max(0.0, 0.3 - center_dev)  # bonus if near center

        total = score + centered_bonus
        if total >= 0.1:  # tune this threshold
            candidates.append({"text": text, "bbox_pdf": (x0, y0, x1, y1), "score": round(total, 3)})

    # sort strongest first
    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates
