from typing import Dict, Any
import re
from sympy.parsing.latex import parse_latex

def _fallback_validate(latex: str) -> Dict[str, Any]:
    latex = (latex or "").strip()
    if not latex:
        return {"ok": True, "errors": []}
    return {"ok": True, "errors": []}

try:
    from equation_scribe.validate import validate_latex as _core_validate
    def validate_latex(latex: str) -> Dict[str, Any]:
        res = _core_validate(latex or "")
        return {"ok": bool(res.ok), "errors": res.errors or []}
except Exception:
    def validate_latex(latex: str) -> Dict[str, Any]:
        return _fallback_validate(latex)
    
def normalize_latex(latex: str) -> str:
    """
    Cleans up common recognition artifacts before validation/display.
    """
    if not latex: return ""

    # 1. Strip 'array' environments (converts to simple lines or just removes wrapper)
    # Matches \begin{array}{...} ... \end{array} across lines
    # This is a naive strip; purely removing the wrapper tags.
    pattern = r"\\begin\{array\}\{.*?\}|\\end\{array\}"
    latex = re.sub(pattern, "", latex, flags=re.DOTALL)

    # 2. Fix common double-escapes if they exist
    latex = latex.replace("\\\\", "\\") 
    
    # 3. Strip leading/trailing whitespace
    return latex.strip()

def validate_latex(latex: str):
    # Use the normalizer first
    clean_latex = normalize_latex(latex)
    
    if not clean_latex:
        return {"ok": False, "errors": ["Empty string"]}
    
    try:
        # Attempt to parse
        parse_latex(clean_latex)
        return {"ok": True, "errors": []}
    except Exception as e:
        # Return the error, but also the normalized version might be useful to see?
        # For now, just return error string
        return {"ok": False, "errors": [str(e)]}