\
from equation_scribe.validate import validate_latex

def test_validate_good_equation():
    res = validate_latex(r"\nabla\cdot \mathbf{E} = \rho/\varepsilon_0")
    assert res.expr is not None
    assert res.canonical_hash is not None

def test_validate_bad_equation():
    res = validate_latex(r"\left( x + y \right.")
    assert not res.ok
    assert res.expr is None
    assert any("left" in e.lower() or "right" in e.lower() for e in res.errors)
