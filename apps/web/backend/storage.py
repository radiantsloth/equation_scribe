from pathlib import Path
from typing import List, Dict, Any
import json
import shutil
from datetime import datetime

from .schemas import EquationRecord

def _maybe_backup_profile_file(profile_dir: Path, fname: str = "equations.jsonl"):
    src = profile_dir / fname
    if not src.exists():
        return None
    history_dir = profile_dir / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    dst = history_dir / f"{fname}.bak.{ts}"
    # copy file preserving metadata
    shutil.copy2(src, dst)
    return dst

def equations_path(root: Path, paper_id: str) -> Path:
    d = root / paper_id
    d.mkdir(parents=True, exist_ok=True)
    return d / "equations.jsonl"


def read_equations(root: Path, paper_id: str) -> List[Dict[str, Any]]:
    p = equations_path(root, paper_id)
    if not p.exists():
        return []
    out: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                # skip broken lines
                continue
            if isinstance(rec, dict):
                out.append(rec)
    return out


def append_equation(root: Path, rec: EquationRecord) -> None:
    p = equations_path(root, rec.paper_id)
    # Pydantic v2 or v1 compatible JSON serialization
    try:
        # v2 path: dump to dict then json.dumps to control ensure_ascii
        payload = json.dumps(rec.model_dump(), ensure_ascii=False)
    except AttributeError:
        # v1 path: use .json with ensure_ascii
        payload = rec.json(ensure_ascii=False)
    with p.open("a", encoding="utf-8") as f:
        f.write(payload + "\n")

def update_equation(root: Path, paper_id: str, eq_uid: str, new_record: Dict[str, Any]) -> None:
    """
    Replace an existing equation record (matching eq_uid) in the JSONL file.
    If no matching eq_uid is found, append the new_record.
    """
    p = equations_path(root, paper_id)
    # backup current state before rewrite
    paper_dir = p.parent
    _maybe_backup_profile_file(paper_dir)
    if not p.exists():
        # Just append if file doesn't exist yet
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(new_record, ensure_ascii=False) + "\n")
        return

    lines = []
    replaced = False
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                lines.append(line)
                continue
            if obj.get("eq_uid") == eq_uid:
                lines.append(json.dumps(new_record, ensure_ascii=False))
                replaced = True
            else:
                lines.append(json.dumps(obj, ensure_ascii=False))

    if not replaced:
        lines.append(json.dumps(new_record, ensure_ascii=False))

    with p.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

def delete_equation(root: Path, paper_id: str, eq_uid: str) -> bool:
    """
    Remove the equation record with eq_uid from the JSONL file for paper_id.
    Returns True if an equation was removed, False if not found.
    """
    p = equations_path(root, paper_id)
    # backup current state before delete
    paper_dir = p.parent
    _maybe_backup_profile_file(paper_dir)
    if not p.exists():
        return False

    lines = []
    removed = False
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                # keep malformed lines unchanged
                lines.append(line)
                continue
            if obj.get("eq_uid") == eq_uid:
                removed = True
                # skip this line (delete)
            else:
                lines.append(json.dumps(obj, ensure_ascii=False))

    if not removed:
        return False

    with p.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return True
