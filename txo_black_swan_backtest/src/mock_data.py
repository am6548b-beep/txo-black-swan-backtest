"""Deterministic mock data for smoke-testing the backtest pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def write_mock_data(data_dir: Path) -> tuple[int, int, int]:
    """Write compact mock market/options/portfolio CSVs that trigger put-spread flow."""

    data_dir.mkdir(parents=True, exist_ok=True)
    dates = pd.bdate_range("2021-01-04", periods=900)
    crash_start = 760
    price = np.full(len(dates), 24_000.0)
    vix = np.full(len(dates), 25.0)
    price[: crash_start - 126] = np.linspace(18_000, 19_000, crash_start - 126)
    price[crash_start - 126 : crash_start] = np.linspace(19_000, 24_000, 126)
    vix[crash_start - 220 : crash_start] = np.linspace(22, 15, 220)
    price[crash_start : crash_start + 10] = np.linspace(price[crash_start - 1], price[crash_start - 1] * 0.78, 10)
    price[crash_start + 10 :] = np.linspace(price[crash_start + 9], price[crash_start + 9] * 1.08, len(dates) - crash_start - 10)
    vix[crash_start : crash_start + 10] = np.linspace(15, 45, 10)
    vix[crash_start + 10 :] = np.linspace(45, 25, len(dates) - crash_start - 10)

    market = pd.DataFrame(
        {
            "date": dates,
            "tx_close": price,
            "tx_open": price,
            "tx_high": price * 1.01,
            "tx_low": price * 0.99,
            "txf_close": price,
            "volume": 100_000,
            "vix": vix,
            "event_flag": 0,
        }
    )

    expiry_calendar = pd.bdate_range(dates.min() + pd.offsets.BDay(30), dates.max() + pd.offsets.BDay(130), freq="20B")
    strikes = list(range(12_000, 34_001, 500))
    rows = []
    for date, spot, day_vix in zip(dates, price, vix):
        expiries = [e for e in expiry_calendar if 20 <= np.busday_count(date.date(), pd.Timestamp(e).date()) <= 125][:3]
        for expiry in expiries:
            dte = int(np.busday_count(date.date(), pd.Timestamp(expiry).date()))
            time_scale = max(dte / 90.0, 0.05)
            for strike in strikes:
                for cp in ("P", "C"):
                    intrinsic = max(strike - spot, 0.0) if cp == "P" else max(spot - strike, 0.0)
                    distance = abs(strike / spot - 1.0)
                    time_value = max(3.0, spot * (day_vix / 100.0) * np.exp(-distance * 9.0) * 0.018 * time_scale)
                    close = intrinsic + time_value
                    spread = 0.08
                    if cp == "P":
                        delta = -max(0.01, min(0.95, 1.0 / (1.0 + np.exp((spot - strike) / 900.0))))
                    else:
                        delta = max(0.01, min(0.95, 1.0 / (1.0 + np.exp((strike - spot) / 900.0))))
                    rows.append(
                        {
                            "date": date,
                            "expiry": expiry,
                            "dte": dte,
                            "cp": cp,
                            "strike": strike,
                            "close": close,
                            "bid": close * (1.0 - spread),
                            "ask": close * (1.0 + spread),
                            "volume": 500,
                            "open_interest": 1_000,
                            "iv": max(0.12, day_vix / 100.0),
                            "delta": delta,
                        }
                    )
    options = pd.DataFrame(rows)

    returns = market["tx_close"].pct_change().fillna(0.0)
    stock_equity = [1_200_000.0]
    beta = 1.3
    for r in returns.iloc[1:]:
        stock_equity.append(stock_equity[-1] * (1.0 + beta * float(r)))
    portfolio = pd.DataFrame({"date": dates, "stock_equity": stock_equity, "portfolio_beta": beta})

    market.to_csv(data_dir / "market.csv", index=False)
    options.to_csv(data_dir / "options.csv", index=False)
    portfolio.to_csv(data_dir / "portfolio.csv", index=False)
    return len(market), len(options), len(portfolio)
