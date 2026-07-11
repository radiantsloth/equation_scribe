from pathlib import Path
import sys

import pytest


CORE_SRC = Path(__file__).resolve().parents[1] / "packages" / "core" / "src"
if str(CORE_SRC) not in sys.path:
    sys.path.insert(0, str(CORE_SRC))

from equation_scribe_core.io import load_index, register_paper, save_index
from equation_scribe_core.models import PaperIndexEntry

def test_save_and_load_index_roundtrip(tmp_path: Path):
    """Saving and loading the core index preserves the paper mapping."""

    root = tmp_path / "profiles"
    idx = load_index(root)
    assert idx.version == 1
    assert idx.papers == {}
    assert idx.by_pdf_basename == {}

    idx.papers["test_paper"] = PaperIndexEntry(
        paper_id="test_paper",
        pdf_basename="test.pdf",
        profiles_dir="test_paper",
        created_at="now",
    )
    idx.by_pdf_basename["test.pdf"] = "test_paper"

    save_index(root, idx)

    loaded = load_index(root)
    assert "test_paper" in loaded.papers
    assert loaded.papers["test_paper"].pdf_basename == "test.pdf"
    assert loaded.by_pdf_basename["test.pdf"] == "test_paper"

def test_register_paper_and_force(tmp_path: Path):
    """register_paper stores normalized values and respects force semantics."""

    root = tmp_path / "profiles"
    paper_id = "p1"
    pdf_basename = "A.PDF"
    register_paper(root, paper_id=paper_id, pdf_basename=pdf_basename, profiles_dir=paper_id, num_equations=2, force=False)

    idx = load_index(root)
    created_at = idx.papers[paper_id].created_at
    assert idx.by_pdf_basename["a.pdf"] == paper_id
    assert idx.papers[paper_id].num_equations == 2
    assert idx.papers[paper_id].pdf_basename == "a.pdf"

    with pytest.raises(RuntimeError):
        register_paper(root, paper_id=paper_id, pdf_basename="a.pdf", profiles_dir=paper_id, num_equations=3, force=False)

    register_paper(root, paper_id=paper_id, pdf_basename="a.pdf", profiles_dir=paper_id, num_equations=5, force=True)
    idx2 = load_index(root)
    assert idx2.papers[paper_id].num_equations == 5
    assert idx2.papers[paper_id].created_at == created_at
