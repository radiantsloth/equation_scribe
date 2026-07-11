"""Shared helpers for the paper-profile directory layout.

These helpers centralize the small filesystem conventions used by the storage
layer. Keeping them in one place makes later JSONL and profile-store migration
steps simpler and keeps path construction readable for newer Python learners.
"""

from pathlib import Path


def ensure_dir(path: Path) -> Path:
    """Create ``path`` if needed and return it.

    Returning the path makes call sites a little easier to read because callers
    can both create and use the directory in one expression.
    """

    path.mkdir(parents=True, exist_ok=True)
    return path


def paper_dir(root: Path, paper_id: str) -> Path:
    """Return the profile directory for one paper under ``root``.

    The directory is created on demand because current storage callers expect
    profile directories to exist before they write JSONL records into them.
    """

    return ensure_dir(Path(root) / paper_id)


def equations_path(root: Path, paper_id: str) -> Path:
    """Return the ``equations.jsonl`` path for one paper profile."""

    return paper_dir(root, paper_id) / "equations.jsonl"


def glossary_path(root: Path, paper_id: str) -> Path:
    """Return the ``glossary.jsonl`` path for one paper profile."""

    return paper_dir(root, paper_id) / "glossary.jsonl"
