"""Configuration loader for `configs/config.yaml`."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = _REPO_ROOT / "configs" / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load project configuration from YAML.

    Defaults to ``configs/config.yaml`` at the repo root. Returns a plain
    dict; missing keys raise ``KeyError`` at the call site so mis-configuration
    fails loudly.
    """
    path = Path(path) if path else _CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def repo_root() -> Path:
    """Absolute path to the repository root (one level above ``src/``)."""
    return _REPO_ROOT
