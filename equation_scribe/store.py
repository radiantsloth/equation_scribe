# equation_scribe/store.py

from __future__ import annotations
import hashlib
from pathlib import Path
import sys
from typing import Dict, Any

try:
    from equation_scribe_core.io import append_jsonl as _core_append_jsonl
    from equation_scribe_core.io import append_equation as _core_append_equation
    from equation_scribe_core.paths import glossary_path
except ModuleNotFoundError:
    core_src = Path(__file__).resolve().parents[1] / "packages" / "core" / "src"
    if str(core_src) not in sys.path:
        sys.path.insert(0, str(core_src))
    from equation_scribe_core.io import append_jsonl as _core_append_jsonl
    from equation_scribe_core.io import append_equation as _core_append_equation
    from equation_scribe_core.paths import glossary_path


def canonical_hash(text: str) -> str:
    """Return the stable short hash used by legacy UI flows."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

def append_jsonl(path: Path, record: Dict[str, Any]):
    """Compatibility wrapper over the shared core JSONL append helper."""

    _core_append_jsonl(path, record)

def save_equation(root: Path, paper_id: str, record: Dict[str, Any]):
    """Compatibility wrapper over the shared core profile-store append helper."""

    _core_append_equation(root, record, paper_id=paper_id)


def save_symbol(root: Path, paper_id: str, record: Dict[str, Any]):
    """Append one glossary record through the shared JSONL helper."""

    append_jsonl(glossary_path(root, paper_id), record)
