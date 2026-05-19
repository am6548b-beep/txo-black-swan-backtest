"""Small shared helpers."""

from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path
from typing import Any


def ensure_dirs(root: Path) -> None:
    for rel in ["data", "reports", "reports/charts"]:
        (root / rel).mkdir(parents=True, exist_ok=True)


def warn(message: str) -> None:
    warnings.warn(message, stacklevel=2)


def load_config_module(path: str | Path) -> Any:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    spec = importlib.util.spec_from_file_location("user_config", config_path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"Cannot load config from {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def year_key(date_like: object) -> int:
    return int(str(date_like)[:4])

