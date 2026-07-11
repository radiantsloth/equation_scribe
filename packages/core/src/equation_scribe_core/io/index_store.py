"""Shared helpers for reading and writing the paper profile index.

This module owns the durable ``index.json`` storage format used to map paper
identifiers to profile directories. The helpers here are intentionally small and
predictable so application layers can reuse the same locking and normalization
behavior.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

import portalocker

from equation_scribe_core.models import PaperIndex, PaperIndexEntry

INDEX_FILENAME = "index.json"


def _now_iso() -> str:
    """Return the current UTC timestamp in the on-disk index format."""

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _empty_index() -> PaperIndex:
    """Build the default empty index structure."""

    return PaperIndex(version=1, papers={}, by_pdf_basename={})


def _coerce_index(index: PaperIndex | Mapping[str, Any]) -> PaperIndex:
    """Normalize either a typed model or a plain mapping into ``PaperIndex``."""

    if isinstance(index, PaperIndex):
        return index
    return PaperIndex.model_validate(index)


def load_index(root: Path) -> PaperIndex:
    """Load ``index.json`` from ``root``.

    The read uses a shared file lock so readers do not race with writers. If the
    file is missing or empty, a default empty index model is returned.
    """

    idx_path = Path(root) / INDEX_FILENAME
    idx_path.parent.mkdir(parents=True, exist_ok=True)

    if not idx_path.exists():
        return _empty_index()

    with idx_path.open("r", encoding="utf-8") as fh:
        portalocker.lock(fh, portalocker.LOCK_SH)
        try:
            fh.seek(0)
            data_text = fh.read()
        finally:
            portalocker.unlock(fh)

    if not data_text.strip():
        return _empty_index()

    data = json.loads(data_text)
    data.setdefault("version", 1)
    data.setdefault("papers", {})
    data.setdefault("by_pdf_basename", {})
    return PaperIndex.model_validate(data)


def save_index(root: Path, index: PaperIndex | Mapping[str, Any]) -> None:
    """Write ``index.json`` atomically under an exclusive lock."""

    model = _coerce_index(index)
    idx_path = Path(root) / INDEX_FILENAME
    idx_path.parent.mkdir(parents=True, exist_ok=True)

    # Serialize first to avoid truncating the file if serialization fails
    serialized_data = json.dumps(model.model_dump(mode="json"), ensure_ascii=False, indent=2)

    mode = "r+" if idx_path.exists() else "w+"
    with idx_path.open(mode, encoding="utf-8") as fh:
        portalocker.lock(fh, portalocker.LOCK_EX)
        try:
            fh.seek(0)
            fh.truncate()
            fh.write(serialized_data)
            fh.flush()
            os.fsync(fh.fileno())
        finally:
            portalocker.unlock(fh)


def register_paper(
    root: Path,
    *,
    paper_id: str,
    pdf_basename: str,
    profiles_dir: Optional[str] = None,
    num_equations: Optional[int] = None,
    force: bool = False,
) -> PaperIndex:
    """Register or update one paper entry in the shared index."""

    profiles_dir = profiles_dir or paper_id
    index = load_index(root)
    papers = index.papers
    by_pdf = index.by_pdf_basename

    normalized_pdf_basename = pdf_basename.lower()
    existing_for_pdf = by_pdf.get(normalized_pdf_basename)
    if existing_for_pdf and existing_for_pdf != paper_id and not force:
        raise RuntimeError(
            f"PDF basename {normalized_pdf_basename!r} is already associated with "
            f"paper_id {existing_for_pdf!r}."
        )

    if paper_id in papers and not force:
        raise RuntimeError(f"paper_id {paper_id!r} already exists in index. Use --force to overwrite.")

    now = _now_iso()
    existing_entry = papers.get(paper_id)
    created_at = existing_entry.created_at if existing_entry and existing_entry.created_at else now

    entry_data: dict[str, Any] = {
        "paper_id": paper_id,
        "pdf_basename": normalized_pdf_basename,
        "profiles_dir": profiles_dir,
        "created_at": created_at,
        "updated_at": now,
    }
    if num_equations is not None:
        entry_data["num_equations"] = int(num_equations)
    elif existing_entry and existing_entry.num_equations is not None:
        entry_data["num_equations"] = existing_entry.num_equations

    papers[paper_id] = PaperIndexEntry.model_validate(entry_data)
    by_pdf[normalized_pdf_basename] = paper_id
    save_index(root, index)
    return index
