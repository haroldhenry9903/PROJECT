"""Resolve paths relative to the repository root (directory containing this file)."""
from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parent


def resolve_repo_path(repo: Path, path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (repo / p)
