#!/usr/bin/env python3
"""
tools/clear_dirs.py

Usage examples:
  # remove detector data/images/synth and annotations file
  python tools/clear_dirs.py --clear-data --data-dir detector/data --yes

  # remove runs with prefix eq_detector under runs/detect
  python tools/clear_dirs.py --clear-runs --runs-dir runs/detect --prefix eq_detector --yes
"""
import argparse
import shutil
import os
import stat
from pathlib import Path
import sys

def _on_rm_error(func, path, exc_info):
    try:
        os.chmod(path, stat.S_IWRITE)
    except Exception:
        pass
    func(path)

def safe_rmtree(path: Path, repo_root: Path):
    path = Path(path).resolve()
    repo_root = Path(repo_root).resolve()
    if not str(path).startswith(str(repo_root)):
        raise RuntimeError(f"Refusing to delete outside repo root: {path}")
    if path == repo_root:
        raise RuntimeError("Refusing to delete repo root")
    if not path.exists():
        return
    shutil.rmtree(path, onerror=_on_rm_error)

def clear_data(data_dir: Path, annotations_file: Path, repo_root: Path, yes=False):
    data_dir = Path(data_dir).resolve()
    annotations_file = Path(annotations_file).resolve()
    print("Will clear data directory:", data_dir)
    print("Will clear annotations file:", annotations_file)
    if not yes:
        if input("Continue? (y/N): ").strip().lower() not in ("y", "yes"):
            print("Aborted.")
            return
    safe_rmtree(data_dir, repo_root)
    if annotations_file.exists():
        print("Deleting annotations file:", annotations_file)
        annotations_file.unlink()
    data_dir.mkdir(parents=True, exist_ok=True)
    print("Done clearing data.")

def clear_runs(runs_dir: Path, prefix: str, repo_root: Path, yes=False):
    runs_dir = Path(runs_dir).resolve()
    if not runs_dir.exists():
        print("Runs directory not found:", runs_dir)
        return
    candidates = [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith(prefix)]
    if not candidates:
        print("No runs found with prefix", prefix)
        return
    print("Will delete these run dirs:")
    for c in candidates:
        print("  ", c)
    if not yes:
        if input("Continue? (y/N): ").strip().lower() not in ("y", "yes"):
            print("Aborted.")
            return
    for c in candidates:
        safe_rmtree(c, repo_root)
    print("Done clearing runs.")

def main():
    repo_root = Path(__file__).resolve().parents[1]  # repo root (tools is under repo root/tools)
    p = argparse.ArgumentParser()
    p.add_argument("--clear-data", action="store_true")
    p.add_argument("--data-dir", default="detector/data", help="Path to data dir to clear")
    p.add_argument("--annotations-file", default="detector/data/annotations/instances_all.json")
    p.add_argument("--clear-runs", action="store_true")
    p.add_argument("--runs-dir", default="runs/detect")
    p.add_argument("--prefix", default="eq_detector", help="prefix for run dirs to delete")
    p.add_argument("-y", "--yes", action="store_true", help="Skip confirmation")
    args = p.parse_args()

    if args.clear_data:
        clear_data(Path(args.data_dir), Path(args.annotations_file), repo_root, yes=args.yes)
    if args.clear_runs:
        clear_runs(Path(args.runs_dir), args.prefix, repo_root, yes=args.yes)

if __name__ == "__main__":
    main()
