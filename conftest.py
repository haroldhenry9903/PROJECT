"""Make repo-root modules importable from the ``tests/`` directory.

Pytest discovers this file at the repo root and inserts the repo path into
``sys.path``, which lets ``tests/test_*.py`` do
``from phase4_smart_optimizer import apply_forecast_noise`` without packaging.
"""
from __future__ import annotations

import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
