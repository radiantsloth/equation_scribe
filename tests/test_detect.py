# tests/test_detect.py
from equation_scribe.detect import find_equation_candidates

def test_find_equation_candidates_basic():
    spans = [
        {"text":"E = mc^2", "bbox_pdf":(100,700,180,720), "page_index":0},
        {"text":"Introduction", "bbox_pdf":(72,720,160,740), "page_index":0},
    ]
    c = find_equation_candidates(spans, page_width=612.0)
    assert c and any("mc^2" in x["text"] for x in c)
