# equation_scribe/tests/test_writeindex.py
import json
import pytest
from pathlib import Path

# Import the functions to test. Adjust import if your package name differs.
from equation_scribe.profile_index import load_index, save_index, register_paper

def test_save_and_load_index_roundtrip(tmp_path: Path):
    """Saving and loading the index returns the same structure."""
    root = tmp_path / "profiles"
    # Start with an empty index
    idx = load_index(root)
    assert isinstance(idx, dict)
    assert "papers" in idx and "by_pdf_basename" in idx

    # Add a paper entry and save
    idx["papers"]["test_paper"] = {
        "paper_id": "test_paper",
        "pdf_basename": "test.pdf",
        "profiles_dir": "test_paper",
        "created_at": "now",
    }
    idx["by_pdf_basename"]["test.pdf"] = "test_paper"

    save_index(root, idx)

    # Load again and assert equality for the part we wrote
    loaded = load_index(root)
    assert "test_paper" in loaded["papers"]
    assert loaded["papers"]["test_paper"]["pdf_basename"] == "test.pdf"
    assert loaded["by_pdf_basename"]["test.pdf"] == "test_paper"

def test_register_paper_and_force(tmp_path: Path):
    """register_paper adds entries; --force allows overwriting."""
    root = tmp_path / "profiles"
    paper_id = "p1"
    pdf_basename = "a.pdf"
    # register the paper
    register_paper(root, paper_id=paper_id, pdf_basename=pdf_basename, profiles_dir=paper_id, num_equations=2, force=False)

    idx = load_index(root)
    assert idx["by_pdf_basename"].get(pdf_basename) == paper_id
    assert idx["papers"][paper_id]["num_equations"] == 2

    # Attempt to register same paper_id without force should raise
    with pytest.raises(RuntimeError):
        register_paper(root, paper_id=paper_id, pdf_basename=pdf_basename, profiles_dir=paper_id, num_equations=3, force=False)

    # With force, should succeed and update num_equations
    register_paper(root, paper_id=paper_id, pdf_basename=pdf_basename, profiles_dir=paper_id, num_equations=5, force=True)
    idx2 = load_index(root)
    assert idx2["papers"][paper_id]["num_equations"] == 5
