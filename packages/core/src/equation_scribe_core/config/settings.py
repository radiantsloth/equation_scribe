"""Shared runtime settings for Equation Scribe applications.

This module centralizes environment-derived filesystem roots so application
layers do not each reimplement their own ``os.getenv(...)`` and directory
creation behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


PROFILES_ROOT_ENV = "PROFILES_ROOT"
PAPERS_ROOT_ENV = "PAPERS_ROOT"
DEFAULT_PROFILES_ROOT = Path("data/profiles")
DEFAULT_PAPERS_ROOT = Path("data/pdfs")


@dataclass(frozen=True)
class RuntimeSettings:
    """Resolved runtime paths for shared storage and source PDFs."""

    profiles_root: Path
    papers_root: Path


def _resolve_path(env_name: str, default: Path) -> Path:
    """Resolve one path from the environment or a default relative path."""

    return Path(os.getenv(env_name, str(default)))


def get_runtime_settings(*, ensure_dirs: bool = True) -> RuntimeSettings:
    """Resolve and optionally create the runtime storage roots."""

    profiles_root = _resolve_path(PROFILES_ROOT_ENV, DEFAULT_PROFILES_ROOT)
    papers_root = _resolve_path(PAPERS_ROOT_ENV, DEFAULT_PAPERS_ROOT)

    if ensure_dirs:
        profiles_root.mkdir(parents=True, exist_ok=True)
        papers_root.mkdir(parents=True, exist_ok=True)

    return RuntimeSettings(profiles_root=profiles_root, papers_root=papers_root)
