from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


def _prepare_real_data_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "prepare_real_data.py"
    spec = importlib.util.spec_from_file_location("prepare_real_data", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_taifex_expiry_supports_friday_weeklies() -> None:
    module = _prepare_real_data_module()

    assert module.parse_taifex_expiry("202507F3") == pd.Timestamp("2025-07-18")
    assert module.parse_taifex_expiry("202507F4") == pd.Timestamp("2025-07-25")


def test_parse_taifex_expiry_supports_wednesday_weeklies_and_monthlies() -> None:
    module = _prepare_real_data_module()

    assert module.parse_taifex_expiry("202501W1") == pd.Timestamp("2025-01-01")
    assert module.parse_taifex_expiry("202401") == pd.Timestamp("2024-01-17")
