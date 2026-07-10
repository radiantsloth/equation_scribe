"""Compatibility wrapper for shared index storage helpers.

The shared index implementation now lives in ``equation_scribe_core``. This
module keeps the legacy import path stable while the rest of the codebase is
migrated.
"""

from pathlib import Path
import sys
from typing import Any, Dict, Optional

try:
    from equation_scribe_core.io import INDEX_FILENAME
    from equation_scribe_core.io import load_index as _core_load_index
    from equation_scribe_core.io import register_paper as _core_register_paper
    from equation_scribe_core.io import save_index as _core_save_index
except ModuleNotFoundError:
    core_src = Path(__file__).resolve().parents[1] / "packages" / "core" / "src"
    if str(core_src) not in sys.path:
        sys.path.insert(0, str(core_src))
    from equation_scribe_core.io import INDEX_FILENAME
    from equation_scribe_core.io import load_index as _core_load_index
    from equation_scribe_core.io import register_paper as _core_register_paper
    from equation_scribe_core.io import save_index as _core_save_index


def load_index(root: Path) -> Dict[str, Any]:
    """Load the shared index and return the legacy dictionary shape."""

    return _core_load_index(root).model_dump(mode="json")


def save_index(root: Path, index: Dict[str, Any]) -> None:
    """Save the legacy dictionary shape through the shared core helper."""

    _core_save_index(root, index)


def register_paper(
    root: Path,
    *,
    paper_id: str,
    pdf_basename: str,
    profiles_dir: Optional[str] = None,
    num_equations: Optional[int] = None,
    force: bool = False,
) -> None:
    """Register one paper entry using the shared core implementation."""

    _core_register_paper(
        root,
        paper_id=paper_id,
        pdf_basename=pdf_basename,
        profiles_dir=profiles_dir,
        num_equations=num_equations,
        force=force,
    )
