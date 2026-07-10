import sys
from pathlib import Path


CORE_SRC = Path(__file__).resolve().parents[1] / "packages" / "core" / "src"
if str(CORE_SRC) not in sys.path:
    sys.path.insert(0, str(CORE_SRC))

from equation_scribe_core.io import JsonlEntry, append_jsonl, read_jsonl, read_jsonl_entries, rewrite_jsonl, write_jsonl
from equation_scribe_core.models import Box, EquationRecord


def test_write_and_read_jsonl_roundtrip(tmp_path: Path):
    path = tmp_path / "equations.jsonl"
    records = [{"eq_uid": "e1", "latex": "x"}, {"eq_uid": "e2", "latex": "y"}]

    write_jsonl(path, records)

    assert read_jsonl(path) == records


def test_append_jsonl_accepts_pydantic_models(tmp_path: Path):
    path = tmp_path / "equations.jsonl"
    record = EquationRecord(
        eq_uid="eq-1",
        paper_id="paper-1",
        latex="E=mc^2",
        notes="",
        boxes=[Box(page=0, bbox_pdf=(1.0, 2.0, 3.0, 4.0))],
    )

    append_jsonl(path, record)

    loaded = read_jsonl(path)
    assert loaded == [
        {
            "eq_uid": "eq-1",
            "paper_id": "paper-1",
            "latex": "E=mc^2",
            "notes": "",
            "boxes": [{"page": 0, "bbox_pdf": [1.0, 2.0, 3.0, 4.0]}],
        }
    ]


def test_read_jsonl_skips_malformed_lines_by_default(tmp_path: Path):
    path = tmp_path / "equations.jsonl"
    path.write_text('{"eq_uid":"good-1"}\nnot-json\n{"eq_uid":"good-2"}\n', encoding="utf-8")

    assert read_jsonl(path) == [{"eq_uid": "good-1"}, {"eq_uid": "good-2"}]

    entries = read_jsonl_entries(path)
    assert entries[1] == JsonlEntry(raw_text="not-json", value=None, is_malformed=True)


def test_rewrite_jsonl_preserves_malformed_lines(tmp_path: Path):
    path = tmp_path / "equations.jsonl"
    path.write_text('{"eq_uid":"keep"}\nnot-json\n{"eq_uid":"drop"}\n', encoding="utf-8")

    rewritten = []
    for entry in read_jsonl_entries(path):
        if entry.is_malformed:
            rewritten.append(entry)
            continue
        if entry.value["eq_uid"] == "keep":
            rewritten.append({"eq_uid": "keep", "latex": "updated"})

    rewrite_jsonl(path, rewritten)

    assert path.read_text(encoding="utf-8") == '{"eq_uid": "keep", "latex": "updated"}\nnot-json\n'
