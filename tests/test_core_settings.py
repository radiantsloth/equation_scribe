import os
import sys
from pathlib import Path


CORE_SRC = Path(__file__).resolve().parents[1] / "packages" / "core" / "src"
if str(CORE_SRC) not in sys.path:
    sys.path.insert(0, str(CORE_SRC))

from equation_scribe_core.config import get_runtime_settings


def test_get_runtime_settings_uses_defaults_and_creates_dirs(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PROFILES_ROOT", raising=False)
    monkeypatch.delenv("PAPERS_ROOT", raising=False)

    settings = get_runtime_settings()

    assert settings.profiles_root == Path("data/profiles")
    assert settings.papers_root == Path("data/pdfs")
    assert (tmp_path / "data/profiles").is_dir()
    assert (tmp_path / "data/pdfs").is_dir()


def test_get_runtime_settings_respects_environment_overrides(tmp_path: Path, monkeypatch):
    profiles_root = tmp_path / "custom-profiles"
    papers_root = tmp_path / "custom-pdfs"
    monkeypatch.setenv("PROFILES_ROOT", str(profiles_root))
    monkeypatch.setenv("PAPERS_ROOT", str(papers_root))

    settings = get_runtime_settings()

    assert settings.profiles_root == profiles_root
    assert settings.papers_root == papers_root
    assert profiles_root.is_dir()
    assert papers_root.is_dir()
