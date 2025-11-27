# detector/tests/conftest.py
# Ensure the project repo root is on sys.path so tests can import the detector package.
import sys
from pathlib import Path

# repo root is two levels up from this file: detector/tests -> detector -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]

repo_root_str = str(REPO_ROOT)
if repo_root_str not in sys.path:
    # Put repo root at front so package imports resolve
    sys.path.insert(0, repo_root_str)
