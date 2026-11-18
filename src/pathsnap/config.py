"""Config loading and repo-relative paths."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


@lru_cache(maxsize=None)
def load_config(name: str) -> dict[str, Any]:
    """Load ``configs/<name>.yaml`` as a dict."""
    path = CONFIG_DIR / f"{name}.yaml"
    with path.open() as fh:
        return yaml.safe_load(fh) or {}


def repo_path(relative: str | Path) -> Path:
    """Resolve a path given relative to the repo root."""
    p = Path(relative)
    return p if p.is_absolute() else REPO_ROOT / p
