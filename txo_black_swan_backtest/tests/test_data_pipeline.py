from __future__ import annotations

import pandas as pd

from src.data_pipeline import build_processed_data
from src.taifex_loader import load_taifex_options


def test_data_pipeline_builds_processed_csvs(tmp_path) -> None:
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    raw.mkdir()
    pd.DataFrame(
        {
            "Date": ["2024-01-02", "2024-01-03"],
            "Open": [100, 101],
            "High": [102, 103],
            "Low": [99, 100],
            "Close": [101, 102],
            "Volume": [1000, 1200],
        }
    ).to_csv(raw / "tx_futures.csv", index=False)
    pd.DataFrame({"Date": ["2024-01-02", "2024-01-03"], "VIX": [15, 16]}).to_csv(raw / "tx_vix.csv", index=False)
    pd.DataFrame(
        {
            "date": ["2024-01-02"],
            "expiry": ["2024-03-01"],
            "cp": ["P"],
            "strike": [9000],
            "close": [10],
            "bid": [9],
            "ask": [11],
            "volume": [100],
            "open_interest": [200],
            "iv": [0.2],
            "delta": [-0.1],
        }
    ).to_csv(raw / "txo_options.csv", index=False)
    pd.DataFrame({"date": ["2024-01-02"], "hbm_asp_index": [100], "ddr5_spot_index": [100]}).to_csv(raw / "macro_factors.csv", index=False)

    result = build_processed_data(raw, processed)

    assert result["market_rows"] == 2
    assert result["options_rows"] == 1
    assert result["macro_rows"] == 1
    assert (processed / "market.csv").exists()
    assert (processed / "options.csv").exists()
    assert (processed / "macro_factors.csv").exists()


def test_taifex_loader_normalizes_call_put_aliases(tmp_path) -> None:
    raw = tmp_path
    pd.DataFrame(
        {
            "交易日期": ["2024-01-02", "2024-01-02"],
            "到期日": ["2024-02-01", "2024-02-01"],
            "買賣權": ["買權", "賣權"],
            "履約價": [10000, 9000],
            "收盤價": [20, 10],
        }
    ).to_csv(raw / "options.csv", index=False)

    options = load_taifex_options(raw)

    assert set(options["cp"]) == {"C", "P"}
    assert {"date", "expiry", "dte", "cp", "strike", "close", "bid", "ask", "volume", "open_interest", "iv", "delta"} <= set(options.columns)
