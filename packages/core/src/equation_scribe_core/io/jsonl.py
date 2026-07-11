"""Shared helpers for reading and writing JSONL files.

These utilities centralize the small JSONL conventions used across the repo.
They support plain JSON-serializable values and Pydantic models, and they make
malformed-line handling explicit so callers can choose between a simple parsed
view and a lossless rewrite path.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Iterable

import portalocker


@dataclass(frozen=True)
class JsonlEntry:
    """One line read from a JSONL file."""

    raw_text: str
    value: Any | None
    is_malformed: bool = False


def _normalize_record(record: Any) -> Any:
    """Convert supported model objects into plain JSON-serializable values."""

    if hasattr(record, "model_dump"):
        try:
            return record.model_dump(mode="json")
        except TypeError:
            return record.model_dump()
    if hasattr(record, "dict"):
        return record.dict()
    return record


def _serialize_record(record: Any) -> str:
    """Serialize one JSONL record to a single line of UTF-8 text."""

    if isinstance(record, JsonlEntry):
        if record.is_malformed:
            return record.raw_text
        record = record.value
    return json.dumps(_normalize_record(record), ensure_ascii=False)


def read_jsonl_entries(path: Path) -> list[JsonlEntry]:
    """Read ``path`` into lossless JSONL entries.

    Blank lines are skipped. Malformed JSON lines are returned with
    ``is_malformed=True`` so callers can preserve the original raw text during
    later rewrites.
    """

    path = Path(path)
    if not path.exists():
        return []

    entries: list[JsonlEntry] = []
    with path.open("r", encoding="utf-8") as fh:
        portalocker.lock(fh, portalocker.LOCK_SH)
        try:
            for raw_line in fh:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    entries.append(JsonlEntry(raw_text=line, value=None, is_malformed=True))
                    continue
                entries.append(JsonlEntry(raw_text=line, value=value, is_malformed=False))
        finally:
            portalocker.unlock(fh)
    return entries


def read_jsonl(path: Path, *, skip_malformed: bool = True) -> list[Any]:
    """Read parsed JSON values from ``path``.

    By default malformed lines are skipped because current storage callers
    already treat them as best-effort data. Set ``skip_malformed=False`` to
    raise on the first malformed line instead.
    """

    records: list[Any] = []
    for entry in read_jsonl_entries(path):
        if entry.is_malformed:
            if skip_malformed:
                continue
            raise ValueError(f"Malformed JSONL line in {path}: {entry.raw_text}")
        records.append(entry.value)
    return records


def append_jsonl(path: Path, record: Any) -> None:
    """Append one record to ``path`` under an exclusive lock."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as fh:
        portalocker.lock(fh, portalocker.LOCK_EX)
        try:
            fh.write(_serialize_record(record) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        finally:
            portalocker.unlock(fh)


def write_jsonl(path: Path, records: Iterable[Any]) -> None:
    """Overwrite ``path`` with the provided JSONL records."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Serialize all records first to prevent partial/corrupted writes if serialization fails
    serialized_lines = [_serialize_record(r) + "\n" for r in records]

    mode = "r+" if path.exists() else "w+"
    with path.open(mode, encoding="utf-8") as fh:
        portalocker.lock(fh, portalocker.LOCK_EX)
        try:
            fh.seek(0)
            fh.truncate()
            fh.writelines(serialized_lines)
            fh.flush()
            os.fsync(fh.fileno())
        finally:
            portalocker.unlock(fh)


def rewrite_jsonl(path: Path, records: Iterable[Any]) -> None:
    """Rewrite ``path`` with updated records using shared JSONL serialization."""

    write_jsonl(path, records)
