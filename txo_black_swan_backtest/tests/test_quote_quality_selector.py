from __future__ import annotations

import pandas as pd

from src.data_loader import load_market, load_options
from src.strategies import ContractSelector


def _config() -> dict:
    return {
        "estimated_spread": 0.08,
        "risk_free_rate": 0.015,
        "wide_spread_volume_threshold": 50,
        "min_open_interest": 100,
    }


def _write_market(data_dir) -> None:
    pd.DataFrame(
        {
            "date": ["2024-01-02"],
            "tx_close": [10000],
            "txf_close": [10000],
            "vix": [20],
            "event_flag": [0],
        }
    ).to_csv(data_dir / "market.csv", index=False)


def test_is_tradable_quote_false_contract_is_not_selected(tmp_path) -> None:
    _write_market(tmp_path)
    pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-02"],
            "expiry": ["2024-03-01", "2024-03-01"],
            "dte": [59, 59],
            "cp": ["P", "P"],
            "strike": [9000, 8900],
            "close": [100, 90],
            "bid": [95, 85],
            "ask": [105, 95],
            "volume": [500, 500],
            "open_interest": [1000, 1000],
            "iv": [0.2, 0.2],
            "delta": [-0.1, -0.1],
            "quote_quality_status": ["ZERO_BID", "VALID"],
            "spread_pct": [None, 0.1],
            "is_tradable_quote": [False, True],
        }
    ).to_csv(tmp_path / "options.csv", index=False)

    market = load_market(tmp_path)
    options = load_options(tmp_path, market, _config())
    selected = ContractSelector(options, _config()).nearest_strike(pd.Timestamp("2024-01-02"), "P", 9000, 30, 90)

    assert selected is not None
    assert selected.strike == 8900
    assert selected.quote_quality_status == "VALID"


def test_valid_quote_still_requires_liquidity(tmp_path) -> None:
    _write_market(tmp_path)
    pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-02"],
            "expiry": ["2024-03-01", "2024-03-01"],
            "dte": [59, 59],
            "cp": ["P", "P"],
            "strike": [9000, 8900],
            "close": [100, 90],
            "bid": [95, 85],
            "ask": [105, 95],
            "volume": [10, 500],
            "open_interest": [1000, 1000],
            "iv": [0.2, 0.2],
            "delta": [-0.1, -0.1],
            "quote_quality_status": ["VALID", "VALID"],
            "spread_pct": [0.1, 0.1],
            "is_tradable_quote": [True, True],
        }
    ).to_csv(tmp_path / "options.csv", index=False)

    market = load_market(tmp_path)
    options = load_options(tmp_path, market, _config())
    selected = ContractSelector(options, _config()).nearest_strike(pd.Timestamp("2024-01-02"), "P", 9000, 30, 90)

    assert selected is not None
    assert selected.strike == 8900


def test_missing_quote_quality_columns_uses_legacy_logic(tmp_path) -> None:
    _write_market(tmp_path)
    pd.DataFrame(
        {
            "date": ["2024-01-02"],
            "expiry": ["2024-03-01"],
            "dte": [59],
            "cp": ["P"],
            "strike": [9000],
            "close": [100],
            "bid": [95],
            "ask": [105],
            "volume": [500],
            "open_interest": [1000],
            "iv": [0.2],
            "delta": [-0.1],
        }
    ).to_csv(tmp_path / "options.csv", index=False)

    market = load_market(tmp_path)
    options = load_options(tmp_path, market, _config())
    selected = ContractSelector(options, _config()).nearest_strike(pd.Timestamp("2024-01-02"), "P", 9000, 30, 90)

    assert selected is not None
    assert selected.strike == 9000
    assert selected.is_tradable_quote is True


def test_by_key_requires_matching_expiry_when_same_strike_exists(tmp_path) -> None:
    _write_market(tmp_path)
    pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-02"],
            "expiry": ["2024-02-01", "2024-03-01"],
            "dte": [30, 59],
            "cp": ["P", "P"],
            "strike": [9000, 9000],
            "close": [50, 100],
            "bid": [45, 95],
            "ask": [55, 105],
            "volume": [500, 500],
            "open_interest": [1000, 1000],
            "iv": [0.2, 0.25],
            "delta": [-0.1, -0.2],
            "quote_quality_status": ["VALID", "VALID"],
            "spread_pct": [0.1, 0.1],
            "is_tradable_quote": [True, True],
        }
    ).to_csv(tmp_path / "options.csv", index=False)

    market = load_market(tmp_path)
    options = load_options(tmp_path, market, _config())
    selector = ContractSelector(options, _config())
    selected = selector.by_key(pd.Timestamp("2024-01-02"), "2024-03-01", "P", 9000)

    assert selected is not None
    assert selected.expiry == "2024-03-01"
    assert selected.close == 100
