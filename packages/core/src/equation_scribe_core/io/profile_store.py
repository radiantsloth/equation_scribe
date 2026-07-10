"""Shared persistence helpers for paper-profile equation records.

This module owns the durable ``equations.jsonl`` storage behavior used by the
web backend and root package. It keeps the on-disk format unchanged while
centralizing read, append, update, delete, and optional history-backup logic in
one place.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import shutil
from typing import Any, Mapping

from equation_scribe_core.io.jsonl import JsonlEntry, append_jsonl, read_jsonl, read_jsonl_entries, rewrite_jsonl
from equation_scribe_core.paths import equations_path


def _record_paper_id(record: Any) -> str | None:
    """Return the ``paper_id`` embedded in a record when available."""

    if hasattr(record, "paper_id"):
        value = getattr(record, "paper_id")
        return str(value) if value is not None else None
    if isinstance(record, Mapping):
        value = record.get("paper_id")
        return str(value) if value is not None else None
    return None


def _resolved_paper_id(record: Any, paper_id: str | None = None) -> str:
    """Choose the explicit or embedded paper identifier for one record."""

    resolved = paper_id or _record_paper_id(record)
    if not resolved:
        raise ValueError("paper_id is required when the record does not include one.")
    return resolved


def backup_profile_file(profile_dir: Path, fname: str = "equations.jsonl") -> Path | None:
    """Create a timestamped history backup for one profile file when it exists."""

    src = Path(profile_dir) / fname
    if not src.exists():
        return None
    history_dir = Path(profile_dir) / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dst = history_dir / f"{fname}.bak.{ts}"
    shutil.copy2(src, dst)
    return dst


def read_equations(root: Path, paper_id: str) -> list[dict[str, Any]]:
    """Read parsed equation records for one paper.

    Malformed JSONL lines are skipped, matching the current best-effort read
    behavior used by the backend.
    """

    path = equations_path(root, paper_id)
    return [record for record in read_jsonl(path) if isinstance(record, dict)]


def append_equation(root: Path, record: Any, *, paper_id: str | None = None) -> Path:
    """Append one equation record and return the target file path."""

    resolved_paper_id = _resolved_paper_id(record, paper_id=paper_id)
    path = equations_path(root, resolved_paper_id)
    append_jsonl(path, record)
    return path


def update_equation(root: Path, paper_id: str, eq_uid: str, new_record: Mapping[str, Any]) -> Path:
    """Replace or append one equation record by ``eq_uid``.

    Existing malformed JSONL lines are preserved verbatim during the rewrite.
    A history backup is created before the file is rewritten when the file
    already exists.
    """

    path = equations_path(root, paper_id)
    backup_profile_file(path.parent)
    if not path.exists():
        append_jsonl(path, dict(new_record))
        return path

    lines: list[JsonlEntry | dict[str, Any] | Any] = []
    replaced = False
    for entry in read_jsonl_entries(path):
        if entry.is_malformed:
            lines.append(entry)
            continue
        obj = entry.value
        if isinstance(obj, dict) and obj.get("eq_uid") == eq_uid:
            lines.append(dict(new_record))
            replaced = True
        else:
            lines.append(obj)

    if not replaced:
        lines.append(dict(new_record))

    rewrite_jsonl(path, lines)
    return path


def delete_equation(root: Path, paper_id: str, eq_uid: str) -> bool:
    """Delete one equation record by ``eq_uid``.

    Returns ``True`` when a matching record was removed. Malformed JSONL lines
    are preserved verbatim if the file must be rewritten.
    """

    path = equations_path(root, paper_id)
    backup_profile_file(path.parent)
    if not path.exists():
        return False

    lines: list[JsonlEntry | dict[str, Any] | Any] = []
    removed = False
    for entry in read_jsonl_entries(path):
        if entry.is_malformed:
            lines.append(entry)
            continue
        obj = entry.value
        if isinstance(obj, dict) and obj.get("eq_uid") == eq_uid:
            removed = True
            continue
        lines.append(obj)

    if not removed:
        return False

    rewrite_jsonl(path, lines)
    return True
