import sys
from pathlib import Path


CORE_SRC = Path(__file__).resolve().parents[1] / "packages" / "core" / "src"
if str(CORE_SRC) not in sys.path:
    sys.path.insert(0, str(CORE_SRC))

from equation_scribe_core.paths import ensure_dir, equations_path, glossary_path, paper_dir


def test_paper_dir_creates_profile_directory(tmp_path: Path):
    root = tmp_path / "profiles"

    path = paper_dir(root, "paper-1")

    assert path == root / "paper-1"
    assert path.is_dir()


def test_equation_and_glossary_paths_share_paper_directory(tmp_path: Path):
    root = tmp_path / "profiles"

    eq_path = equations_path(root, "paper-1")
    gl_path = glossary_path(root, "paper-1")

    assert eq_path == root / "paper-1" / "equations.jsonl"
    assert gl_path == root / "paper-1" / "glossary.jsonl"
    assert (root / "paper-1").is_dir()


def test_ensure_dir_returns_created_path(tmp_path: Path):
    target = tmp_path / "nested" / "folder"

    returned = ensure_dir(target)

    assert returned == target
    assert target.is_dir()
