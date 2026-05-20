"""Data directory resolution helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def ensure_data_layout(root: Path) -> None:
    for rel in ["data/raw", "data/raw/taifex", "data/raw/macro", "data/processed", "data/sample"]:
        (root / rel).mkdir(parents=True, exist_ok=True)


def has_required_processed_data(data_dir: Path) -> bool:
    required = ["market.csv", "options.csv"]
    for name in required:
        path = data_dir / name
        if not path.exists() or path.stat().st_size == 0:
            return False
        try:
            if pd.read_csv(path, nrows=1).empty:
                return False
        except Exception:
            return False
    return True


def resolve_runtime_data_dir(root: Path, config: dict) -> Path:
    configured = config.get("data_dir")
    if configured:
        path = Path(configured)
        return path if path.is_absolute() else root / path
    processed = root / "data" / "processed"
    if has_required_processed_data(processed):
        return processed
    return root / "data" / "sample"


def resolve_processed_data_dir(root: Path, config: dict) -> Path:
    configured = config.get("processed_data_dir", "data/processed")
    path = Path(configured)
    return path if path.is_absolute() else root / path


def resolve_raw_data_dir(root: Path, config: dict) -> Path:
    configured = config.get("raw_data_dir", "data/raw")
    path = Path(configured)
    return path if path.is_absolute() else root / path
