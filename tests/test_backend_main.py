import importlib
import sys
import types
from pathlib import Path

import pytest
from fastapi import HTTPException


ROOT = Path(__file__).resolve().parents[1]
CORE_SRC = ROOT / "packages" / "core" / "src"
if str(CORE_SRC) not in sys.path:
    sys.path.insert(0, str(CORE_SRC))


@pytest.fixture
def backend_main(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delitem(sys.modules, "apps.web.backend.main", raising=False)
    monkeypatch.setenv("PROFILES_ROOT", "data/profiles")
    monkeypatch.setenv("PAPERS_ROOT", "data/pdfs")

    recognition_module = types.ModuleType("equation_scribe.recognition.inference")
    recognition_module.image_to_latex = lambda image: "stub-latex"
    monkeypatch.setitem(sys.modules, "equation_scribe.recognition.inference", recognition_module)

    detector_module = types.ModuleType("equation_scribe.detector.inference")
    detector_module.detect_image = lambda *args, **kwargs: []
    monkeypatch.setitem(sys.modules, "equation_scribe.detector.inference", detector_module)

    return importlib.import_module("apps.web.backend.main")


def test_backend_main_uses_shared_runtime_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.delitem(sys.modules, "apps.web.backend.main", raising=False)
    monkeypatch.setenv("PROFILES_ROOT", str(tmp_path / "profiles-env"))
    monkeypatch.setenv("PAPERS_ROOT", str(tmp_path / "papers-env"))

    recognition_module = types.ModuleType("equation_scribe.recognition.inference")
    recognition_module.image_to_latex = lambda image: "stub-latex"
    monkeypatch.setitem(sys.modules, "equation_scribe.recognition.inference", recognition_module)

    detector_module = types.ModuleType("equation_scribe.detector.inference")
    detector_module.detect_image = lambda *args, **kwargs: []
    monkeypatch.setitem(sys.modules, "equation_scribe.detector.inference", detector_module)

    backend_main = importlib.import_module("apps.web.backend.main")

    assert backend_main.PROFILES_ROOT == tmp_path / "profiles-env"
    assert backend_main.PAPERS_ROOT == tmp_path / "papers-env"
    assert backend_main.PROFILES_ROOT.is_dir()
    assert backend_main.PAPERS_ROOT.is_dir()


def test_get_profiles_index_endpoint_uses_core_index_store(tmp_path: Path, backend_main):
    backend_main.PROFILES_ROOT = tmp_path / "profiles"

    result = backend_main.get_profiles_index_endpoint()

    assert result == {"version": 1, "papers": {}, "by_pdf_basename": {}}


def test_list_equations_reads_from_core_profile_store(tmp_path: Path, backend_main):
    backend_main.PROFILES_ROOT = tmp_path / "profiles"

    record = backend_main.EquationRecord(
        eq_uid="eq-1",
        paper_id="paper-1",
        latex="x",
        notes="",
        boxes=[{"page": 0, "bbox_pdf": (1.0, 2.0, 3.0, 4.0)}],
    )
    backend_main.append_equation(backend_main.PROFILES_ROOT, record)

    result = backend_main.list_equations("paper-1")

    assert result == {
        "items": [
            {
                "eq_uid": "eq-1",
                "paper_id": "paper-1",
                "latex": "x",
                "notes": "",
                "boxes": [{"page": 0, "bbox_pdf": [1.0, 2.0, 3.0, 4.0]}],
            }
        ]
    }


def test_save_update_and_delete_equation_endpoints_use_core_store(tmp_path: Path, backend_main):
    backend_main.PROFILES_ROOT = tmp_path / "profiles"
    backend_main._adjudicate_record = lambda paper_id, rec: None

    record = backend_main.EquationRecord(
        eq_uid="eq-1",
        paper_id="paper-1",
        latex="x",
        notes="",
        boxes=[{"page": 0, "bbox_pdf": (1.0, 2.0, 3.0, 4.0)}],
    )

    assert backend_main.save_equation("paper-1", record) == {"ok": True}
    assert backend_main.list_equations("paper-1")["items"][0]["latex"] == "x"

    updated = backend_main.EquationRecord(
        eq_uid="eq-1",
        paper_id="paper-1",
        latex="y",
        notes="updated",
        boxes=[{"page": 0, "bbox_pdf": (1.0, 2.0, 3.0, 4.0)}],
    )
    assert backend_main.update_equation_endpoint("paper-1", "eq-1", updated) == {"ok": True}
    assert backend_main.list_equations("paper-1")["items"][0]["latex"] == "y"

    assert backend_main.delete_equation_endpoint("paper-1", "eq-1") == {"ok": True}
    assert backend_main.list_equations("paper-1") == {"items": []}


def test_delete_equation_endpoint_raises_when_missing(tmp_path: Path, backend_main):
    backend_main.PROFILES_ROOT = tmp_path / "profiles"

    try:
        backend_main.delete_equation_endpoint("paper-1", "missing")
    except HTTPException as exc:
        assert exc.status_code == 404
        assert exc.detail == "Equation not found"
    else:
        raise AssertionError("Expected HTTPException for missing equation")
