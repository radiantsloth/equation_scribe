# tests/test_detect_columns.py
import pytest
from equation_scribe.detect import cluster_spans_into_lines

def test_column_separation():
    # Mock spans: Two columns.
    # Col 1: "The" (x=10) ... "Energy" (x=50)
    # Col 2: "E = mc^2" (x=300) ... (x=350)
    # They are at the same Y level (y=100)
    
    y = 100.0
    col1_span = {"text": "Energy", "bbox_pdf": (10.0, y, 50.0, y+10)}
    col2_span = {"text": "E=mc^2", "bbox_pdf": (300.0, y, 350.0, y+10)} # 250px gap
    
    spans = [col1_span, col2_span]
    
    # Run clustering
    segments = cluster_spans_into_lines(spans, x_gap_threshold=50.0)
    
    # Should result in 2 separate segments, not 1
    assert len(segments) == 2
    assert segments[0][0]["text"] == "Energy"
    assert segments[1][0]["text"] == "E=mc^2"
    
def test_no_separation_for_close_words():
    # "Hello" (x=10..40) "World" (x=45..80). Gap is 5. Should merge.
    y = 100.0
    w1 = {"text": "Hello", "bbox_pdf": (10, y, 40, y+10)}
    w2 = {"text": "World", "bbox_pdf": (45, y, 80, y+10)}
    
    spans = [w1, w2]
    segments = cluster_spans_into_lines(spans, x_gap_threshold=50.0)
    
    assert len(segments) == 1
    assert len(segments[0]) == 2