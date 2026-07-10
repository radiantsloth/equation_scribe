"""Repo-level pytest configuration.

Pytest temp directory handling is unreliable in this Windows setup:

- a fixed shared ``--basetemp`` can fail when another process still holds a
  handle open
- the default temp-root logic scans shared parent folders such as
  ``pytest-of-<user>``, which has also produced ``PermissionError`` in this
  environment

Give each pytest process its own unique repo-local base temp directory so temp
fixtures never contend over one shared root.
"""

from __future__ import annotations

from pathlib import Path
import os
import uuid

import pytest

_PYTEST_TEMP_ROOT = Path(__file__).resolve().parent / ".pytest-tmp-root"
_PYTEST_TEMP_ROOT.mkdir(parents=True, exist_ok=True)


def pytest_configure(config: pytest.Config) -> None:
    """Force one unique repo-local ``basetemp`` per pytest process."""

    if config.option.basetemp:
        return
    run_id = f"run-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    config.option.basetemp = str(_PYTEST_TEMP_ROOT / run_id)
