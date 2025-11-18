# equation_scribe/profile_index.py
from pathlib import Path
from typing import Dict, Any, Optional
import json
from datetime import datetime, timezone

INDEX_FILENAME = "index.json"

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def load_index(root: Path) -> Dict[str, Any]:
    idx = root / INDEX_FILENAME
    if not idx.exists():
        return {"version": 1, "papers": {}, "by_pdf_basename": {}}
    with idx.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "version" not in data:
        data["version"] = 1
    if "papers" not in data:
        data["papers"] = {}
    if "by_pdf_basename" not in data:
        data["by_pdf_basename"] = {}
    return data

def save_index(root: Path, index: Dict[str, Any]) -> None:
    idx = root / INDEX_FILENAME
    tmp = idx.with_suffix(".tmp")
    root.mkdir(parents=True, exist_ok=True)
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    tmp.replace(idx)

def register_paper(
    root: Path,
    *,
    paper_id: str,
    pdf_basename: str,
    profiles_dir: Optional[str] = None,
    num_equations: Optional[int] = None,
    force: bool = False,
) -> None:
    profiles_dir = profiles_dir or paper_id
    index = load_index(root)
    papers = index["papers"]
    by_pdf = index["by_pdf_basename"]

    existing_for_pdf = by_pdf.get(pdf_basename)
    if existing_for_pdf and existing_for_pdf != paper_id and not force:
        raise RuntimeError(
            f"PDF basename {pdf_basename!r} is already associated with paper_id {existing_for_pdf!r}."
        )

    if paper_id in papers and not force:
        raise RuntimeError(f"paper_id {paper_id!r} already exists in index. Use --force to overwrite.")

    now = _now_iso()
    entry = papers.get(paper_id, {})
    created_at = entry.get("created_at", now)

    entry.update(
        {
            "paper_id": paper_id,
            "pdf_basename": pdf_basename,
            "profiles_dir": profiles_dir,
            "created_at": created_at,
            "updated_at": now,
        }
    )

    if num_equations is not None:
        entry["num_equations"] = int(num_equations)

    papers[paper_id] = entry
    by_pdf[pdf_basename] = paper_id

    save_index(root, index)
