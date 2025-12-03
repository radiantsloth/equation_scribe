#!/usr/bin/env python3
"""
tools/safe_import_rewriter.py

Rewrite imports:
  - `import detector`                -> `import equation_scribe.detector as detector`
  - `import detector as d`           -> `import equation_scribe.detector as d`
  - `import detector.foo`            -> `import equation_scribe.detector.foo`
  - `from detector import X`         -> `from equation_scribe.detector import X`
  - `from detector.foo import X`     -> `from equation_scribe.detector.foo import X`

The script updates .py files in place and writes a .bak backup for safety.

Usage:
  python tools/safe_import_rewriter.py --root . --backup
"""
import ast
import argparse
from pathlib import Path
import io
import sys

SKIP_DIRS = {".git", "venv", "env", "__pycache__", "node_modules", "frontend/node_modules"}

def rewrite_file(path: Path, dry_run=False, backup=True):
    src = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(src)
    except SyntaxError:
        print(f"Skipping (syntax error): {path}")
        return False

    # Collect replacements: list of (old_segment, new_segment)
    replacements = []

    for node in ast.walk(tree):
        # handle Import nodes
        if isinstance(node, ast.Import):
            # reconstruct the original text segment for the node
            try:
                old_seg = ast.get_source_segment(src, node)
            except Exception:
                old_seg = None
            new_parts = []
            changed = False
            for alias in node.names:
                name = alias.name  # e.g., 'detector' or 'detector.module'
                asname = alias.asname
                if name == "detector":
                    newname = "equation_scribe.detector"
                    changed = True
                elif name.startswith("detector."):
                    newname = "equation_scribe." + name
                    changed = True
                else:
                    newname = name
                if asname:
                    new_parts.append(f"{newname} as {asname}")
                else:
                    new_parts.append(newname)
            if changed and old_seg:
                new_seg = "import " + ", ".join(new_parts)
                replacements.append((old_seg, new_seg))

        # handle ImportFrom nodes
        elif isinstance(node, ast.ImportFrom):
            # skip relative imports (levels > 0)
            if node.level and node.level > 0:
                continue
            module = node.module  # e.g., 'detector' or 'detector.foo'
            if not module:
                continue
            if module == "detector" or module.startswith("detector."):
                try:
                    old_seg = ast.get_source_segment(src, node)
                except Exception:
                    old_seg = None
                # build new module
                new_module = "equation_scribe." + module
                # reconstruct names with as
                names_txt = []
                for alias in node.names:
                    if alias.asname:
                        names_txt.append(f"{alias.name} as {alias.asname}")
                    else:
                        names_txt.append(alias.name)
                new_seg = f"from {new_module} import " + ", ".join(names_txt)
                if old_seg:
                    replacements.append((old_seg, new_seg))

    if not replacements:
        return False

    new_src = src
    # Apply replacements; ensure we replace unique texts (do left->right)
    for old, new in replacements:
        if old not in new_src:
            # As a fallback, try to replace normalized whitespace variant
            # but generally old should be present.
            print(f"Warning: expected segment not found in {path}: {old!r}")
            continue
        new_src = new_src.replace(old, new, 1)

    if new_src == src:
        return False

    if dry_run:
        print(f"[DRY] Would update: {path}")
        for old, new in replacements:
            print("  -", old.strip(), "->", new.strip())
        return True

    if backup:
        bak = path.with_suffix(path.suffix + ".bak")
        bak.write_text(src, encoding="utf-8")

    path.write_text(new_src, encoding="utf-8")
    print(f"Updated: {path}")
    return True

def iter_py_files(root: Path):
    for p in root.rglob("*.py"):
        # skip vendor/virtual environ directories
        parts = set(p.parts)
        if parts & SKIP_DIRS:
            continue
        yield p

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", help="Repo root")
    parser.add_argument("--dry-run", action="store_true", help="Show changes but don't write")
    parser.add_argument("--backup", action="store_true", default=True, help="Write .bak backups")
    args = parser.parse_args()

    root = Path(args.root)
    changed_files = []
    for p in iter_py_files(root):
        changed = rewrite_file(p, dry_run=args.dry_run, backup=args.backup)
        if changed:
            changed_files.append(p)
    print(f"Done. Modified {len(changed_files)} files.")
    if args.dry_run:
        print("Dry run; no files were written.")
    else:
        print("Backups (.bak) written for modified files.")

if __name__ == "__main__":
    main()
