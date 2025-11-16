# equation_scribe/store.py
from __future__ import annotations
import json, hashlib
from pathlib import Path
from typing import Dict, Any

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def paper_dir(root: Path, paper_id: str) -> Path:
    d = root / paper_id
    ensure_dir(d)
    return d

def canonical_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

def append_jsonl(path: Path, record: Dict[str, Any]):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

# def save_equation(root: Path, paper_id: str, record: Dict[str, Any]):
#     d = paper_dir(root, paper_id)
#     append_jsonl(d / "equations.jsonl", record)
def save_equation(root: Path, paper_id: str, record: Dict[str, Any]):
    d = paper_dir(root, paper_id)
    path = d / "equations.jsonl"
    print(f"[save_equation] writing to: {path}")  # DEBUG
    append_jsonl(path, record)


def save_symbol(root: Path, paper_id: str, record: Dict[str, Any]):
    d = paper_dir(root, paper_id)
    append_jsonl(d / "glossary.jsonl", record)
