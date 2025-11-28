"""
Compatibility shim that makes `import detector` keep working after moving
the real detector code to `equation_scribe/detector`.

This module sets the top-level detector package __path__ to the
equation_scribe.detector package __path__, so submodule imports like
`import detector.render_latex` will find modules under
`equation_scribe/detector/...`.

If for some reason the subpackage is not importable yet, the shim falls
back to a no-op so errors arise later when trying to import submodules.
"""
from importlib import import_module

try:
    subpkg = import_module("equation_scribe.detector")
    # Delegate the package path to the real location
    __path__ = list(subpkg.__path__)
except Exception:
    # If import fails, leave __path__ alone (imports will error as before).
    # We don't want to mask errors here, but this keeps this shim importable.
    __path__ = []
