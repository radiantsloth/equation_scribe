\
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Tuple
import hashlib
import re

from sympy import simplify, srepr

# Try to import SymPy's LaTeX parser; provide a helpful error if missing
try:
    from sympy.parsing.latex import parse_latex
    _HAS_PARSE_LATEX = True
    _PARSE_ERR = None
except Exception as e:
    parse_latex = None
    _HAS_PARSE_LATEX = False
    _PARSE_ERR = e

@dataclass
class ValidationResult:
    ok: bool
    errors: List[str]
    expr: Optional[object]
    canonical_hash: Optional[str]

_BRACE_PAIRS = {"{": "}", "(": ")", "[": "]"}

def _balanced_braces(s: str) -> Tuple[bool, str]:
    """Return (ok, message) after a simple brace/paren/ bracket balance check."""
    stack = []
    for i, ch in enumerate(s):
        if ch in _BRACE_PAIRS:
            stack.append(_BRACE_PAIRS[ch])
        elif ch in _BRACE_PAIRS.values():
            if not stack or stack.pop() != ch:
                return False, f"Unmatched closing '{ch}' at pos {i}"
    if stack:
        return False, "Unclosed brace(s) at end"
    return True, ""

def _micro_normalize(latex: str) -> str:
    """Small cleanups that reduce trivial variation but preserve math meaning."""
    t = latex.strip()
    # remove \( \), \[ \] wrappers if present
    t = re.sub(r"^\\\(|\\\)$", "", t)
    t = re.sub(r"^\\\[|\\\]$", "", t)
    # collapse multiple spaces
    t = re.sub(r"\s+", " ", t)
    return t

def _canonical_hash_from_expr(expr) -> str:
    """Stable short hash from a canonical SymPy string form."""
    try:
        can = srepr(simplify(expr))
    except Exception:
        can = srepr(expr)
    return hashlib.sha256(can.encode("utf-8")).hexdigest()[:16]

def validate_latex(latex: str) -> ValidationResult:
    """
    Validate LaTeX, attempt to parse to a SymPy object, and compute a canonical hash.
    Returns ValidationResult(ok, errors, expr, canonical_hash).
    """
    errs: List[str] = []
    if not latex or not latex.strip():
        return ValidationResult(False, ["Empty LaTeX"], None, None)

    s = _micro_normalize(latex)

    ok, msg = _balanced_braces(s)
    if not ok:
        errs.append(f"Brace check: {msg}")

    if "\\left" in s and "\\right" not in s:
        errs.append("Has \\left but missing matching \\right")
    if "\\right" in s and "\\left" not in s:
        errs.append("Has \\right but missing matching \\left")

    if not _HAS_PARSE_LATEX:
        errs.append(
            f"LaTeX parser unavailable: {_PARSE_ERR}. "
            "Install the ANTLR4 runtime (pip: 'pip install antlr4-python3-runtime==4.11')."
        )
        return ValidationResult(False, errs, None, None)

    expr = None
    try:
        expr = parse_latex(s)
    except Exception as e:
        errs.append(f"SymPy parse error: {type(e).__name__}: {e}")
        return ValidationResult(False, errs, None, None)

    chash = _canonical_hash_from_expr(expr)
    return ValidationResult(len(errs) == 0, errs, expr, chash)
