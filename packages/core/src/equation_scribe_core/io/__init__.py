"""I/O helpers for Equation Scribe core."""

from .index_store import INDEX_FILENAME, load_index, register_paper, save_index
from .jsonl import JsonlEntry, append_jsonl, read_jsonl, read_jsonl_entries, rewrite_jsonl, write_jsonl
from .profile_store import (
    append_equation,
    backup_profile_file,
    delete_equation,
    read_equations,
    update_equation,
)

__all__ = [
    "INDEX_FILENAME",
    "JsonlEntry",
    "append_jsonl",
    "append_equation",
    "backup_profile_file",
    "delete_equation",
    "load_index",
    "read_jsonl",
    "read_jsonl_entries",
    "read_equations",
    "register_paper",
    "rewrite_jsonl",
    "save_index",
    "update_equation",
    "write_jsonl",
]
