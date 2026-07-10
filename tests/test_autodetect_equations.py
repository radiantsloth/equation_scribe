from pathlib import Path
import sys

from equation_scribe.autodetect_equations import write_detected_equations


CORE_SRC = Path(__file__).resolve().parents[1] / "packages" / "core" / "src"
if str(CORE_SRC) not in sys.path:
    sys.path.insert(0, str(CORE_SRC))

from equation_scribe_core.io import load_index


def test_write_detected_equations_writes_jsonl_and_updates_index(tmp_path: Path):
    root = tmp_path / "profiles"
    records = [
        {
            "eq_uid": "eq-1",
            "paper_id": "paper-1",
            "latex": "x+y",
            "notes": "",
            "boxes": [{"page": 0, "bbox_pdf": [1.0, 2.0, 3.0, 4.0]}],
        }
    ]

    out_path = write_detected_equations(
        records,
        pdf_path="paper.pdf",
        paper_id="paper-1",
        data_root=root,
        force=False,
    )

    assert out_path == root / "paper-1" / "equations.jsonl"
    assert out_path.read_text(encoding="utf-8") == (
        '{"eq_uid": "eq-1", "paper_id": "paper-1", "latex": "x+y", "notes": "", '
        '"boxes": [{"page": 0, "bbox_pdf": [1.0, 2.0, 3.0, 4.0]}]}\n'
    )

    index = load_index(root)
    assert index.papers["paper-1"].pdf_basename == "paper.pdf"
    assert index.papers["paper-1"].num_equations == 1


def test_write_detected_equations_respects_force_and_creates_backup(tmp_path: Path):
    root = tmp_path / "profiles"
    paper_dir = root / "paper-1"
    paper_dir.mkdir(parents=True, exist_ok=True)
    out_path = paper_dir / "equations.jsonl"
    out_path.write_text('{"eq_uid": "old"}\n', encoding="utf-8")

    records = [
        {
            "eq_uid": "eq-2",
            "paper_id": "paper-1",
            "latex": "z",
            "notes": "",
            "boxes": [],
        }
    ]

    skipped = write_detected_equations(
        records,
        pdf_path="paper.pdf",
        paper_id="paper-1",
        data_root=root,
        force=False,
    )
    assert skipped is None
    assert out_path.read_text(encoding="utf-8") == '{"eq_uid": "old"}\n'

    written = write_detected_equations(
        records,
        pdf_path="paper.pdf",
        paper_id="paper-1",
        data_root=root,
        force=True,
    )

    assert written == out_path
    assert out_path.read_text(encoding="utf-8") == (
        '{"eq_uid": "eq-2", "paper_id": "paper-1", "latex": "z", "notes": "", "boxes": []}\n'
    )
    backups = list(paper_dir.glob("equations.jsonl.bak.*"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == '{"eq_uid": "old"}\n'
