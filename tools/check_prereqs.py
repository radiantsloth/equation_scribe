#!/usr/bin/env python3
"""
Simple system prerequisite checker for Equation Scribe.

Checks availability of external binaries (pdflatex, pdftoppm, tesseract, node, npm)
and important Python packages (Pillow, numpy, matplotlib, pdf2image, pytesseract,
ultralytics, portalocker, sympy, pdfplumber, pymupdf, torch). Prints versions
and recommendations for missing items.

Usage:
    python tools/check_prereqs.py
"""
from __future__ import annotations

import subprocess
import sys
import shutil
import importlib
from typing import List, Tuple


BINARIES = ["pdflatex", "pdftoppm", "tesseract", "node", "npm"]
# Note: module names to import - PIL is provided by Pillow package
PY_PACKAGES = [
    ("PIL", "Pillow"),
    ("numpy", "numpy"),
    ("matplotlib", "matplotlib"),
    ("pdf2image", "pdf2image"),
    ("pytesseract", "pytesseract"),
    ("ultralytics", "ultralytics"),
    ("portalocker", "portalocker"),
    ("sympy", "sympy"),
    ("pdfplumber", "pdfplumber"),
    ("fitz", "PyMuPDF"),
    ("torch", "torch"),
]


def check_binary(name: str) -> Tuple[bool, str]:
    path = shutil.which(name)
    if not path:
        return False, "not found on PATH"
    try:
        p = subprocess.run([name, "--version"], capture_output=True, text=True, timeout=6)
        out = p.stdout.strip() or p.stderr.strip()
        first_line = out.splitlines()[0] if out else f"found at {path}"
        return True, first_line
    except Exception as e:
        # some binaries use "-v" or different flags; just report found
        return True, f"found at {path} (version unknown: {e})"


def check_python_package(pkg: str) -> Tuple[bool, str]:
    try:
        spec = importlib.util.find_spec(pkg)
        if spec is None:
            return False, "not installed"
        mod = importlib.import_module(pkg)
        ver = getattr(mod, "__version__", None) or getattr(mod, "VERSION", None) or "version unknown"
        return True, str(ver)
    except Exception as e:
        return False, f"error importing: {e}"


def main():
    print("\n### System prerequisite check for Equation Scribe ###\n")
    print("Checking external binaries...")
    missing_bins: List[str] = []
    for b in BINARIES:
        ok, msg = check_binary(b)
        status = "OK" if ok else "MISSING"
        print(f"  - {b}: {status} -- {msg}")
        if not ok:
            missing_bins.append(b)

    print("\nChecking Python packages...")
    missing_pkgs: List[str] = []
    for import_name, pretty in PY_PACKAGES:
        ok, msg = check_python_package(import_name)
        status = "OK" if ok else "MISSING"
        print(f"  - {pretty} ({import_name}): {status} -- {msg}")
        if not ok:
            missing_pkgs.append(pretty)

    print("\nSummary:")
    if not missing_bins and not missing_pkgs:
        print("  All checked prerequisites appear to be present.")
    else:
        if missing_bins:
            print(f"  Missing binaries: {', '.join(missing_bins)}")
        if missing_pkgs:
            print(f"  Missing Python packages: {', '.join(missing_pkgs)}")

        print("\nRecommendations:")
        print("  - Install MiKTeX (Windows) or TeX Live (Linux) to get pdflatex.")
        print("  - Install poppler (for pdftoppm) for pdf2image conversions.")
        print("  - Install Tesseract OCR for OCR fallback.")
        print("  - Install Node.js and npm if you will run the frontend.")
        print("  - For Python packages, create your environment and run:")
        print("      pip install -r requirements-dev.txt")
        print("  - If you use ultralytics/yolo, ensure the 'yolo' CLI is on PATH or")
        print("    run training via 'python -m ultralytics' or 'python -m ultralytics.yolo'.")

    print("\nNote: Some binaries (e.g. pdflatex) may be installed but not on PATH.\n"
          "Make sure their install location is exported into PATH if the script cannot find them.")


if __name__ == "__main__":
    main()
