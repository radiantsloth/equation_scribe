import sys
from pathlib import Path


CORE_SRC = Path(__file__).resolve().parents[1] / "packages" / "core" / "src"
if str(CORE_SRC) not in sys.path:
    sys.path.insert(0, str(CORE_SRC))

from equation_scribe.store import save_equation
from equation_scribe_core.io import append_equation, delete_equation, read_equations, update_equation
from equation_scribe_core.models import Box, EquationRecord


def test_append_and_read_equations_roundtrip(tmp_path: Path):
    root = tmp_path / "profiles"
    record = EquationRecord(
        eq_uid="eq-1",
        paper_id="paper-1",
        latex="x+y",
        notes="n",
        boxes=[Box(page=0, bbox_pdf=(1.0, 2.0, 3.0, 4.0))],
    )

    append_equation(root, record)

    assert read_equations(root, "paper-1") == [
        {
            "eq_uid": "eq-1",
            "paper_id": "paper-1",
            "latex": "x+y",
            "notes": "n",
            "boxes": [{"page": 0, "bbox_pdf": [1.0, 2.0, 3.0, 4.0]}],
        }
    ]


def test_update_equation_rewrites_matching_record_and_creates_backup(tmp_path: Path):
    root = tmp_path / "profiles"
    append_equation(root, {"eq_uid": "keep", "paper_id": "paper-1", "latex": "old", "boxes": []})
    path = root / "paper-1" / "equations.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        fh.write("not-json\n")

    update_equation(root, "paper-1", "keep", {"eq_uid": "keep", "paper_id": "paper-1", "latex": "new", "boxes": []})

    assert read_equations(root, "paper-1") == [
        {"eq_uid": "keep", "paper_id": "paper-1", "latex": "new", "boxes": []}
    ]
    assert path.read_text(encoding="utf-8") == '{"eq_uid": "keep", "paper_id": "paper-1", "latex": "new", "boxes": []}\nnot-json\n'

    history_dir = root / "paper-1" / "history"
    backups = list(history_dir.glob("equations.jsonl.bak.*"))
    assert len(backups) == 1
    assert '"eq_uid": "keep"' in backups[0].read_text(encoding="utf-8")


def test_delete_equation_removes_record_and_creates_backup(tmp_path: Path):
    root = tmp_path / "profiles"
    append_equation(root, {"eq_uid": "drop", "paper_id": "paper-1", "latex": "x", "boxes": []})
    append_equation(root, {"eq_uid": "keep", "paper_id": "paper-1", "latex": "y", "boxes": []})

    removed = delete_equation(root, "paper-1", "drop")

    assert removed is True
    assert read_equations(root, "paper-1") == [
        {"eq_uid": "keep", "paper_id": "paper-1", "latex": "y", "boxes": []}
    ]
    backups = list((root / "paper-1" / "history").glob("equations.jsonl.bak.*"))
    assert len(backups) == 1


def test_save_equation_legacy_wrapper_accepts_explicit_paper_id(tmp_path: Path):
    root = tmp_path / "profiles"

    save_equation(root, "paper-legacy", {"eq_uid": "legacy-1", "latex": "z", "boxes": []})

    assert read_equations(root, "paper-legacy") == [
        {"eq_uid": "legacy-1", "latex": "z", "boxes": []}
    ]
