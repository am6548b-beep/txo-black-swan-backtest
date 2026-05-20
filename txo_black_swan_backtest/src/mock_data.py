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

    macro = pd.DataFrame(
        {
            "date": dates,
            "hbm_asp_index": np.linspace(100, 180, len(dates)),
            "ddr5_spot_index": np.r_[np.linspace(100, 155, crash_start), np.linspace(155, 110, len(dates) - crash_start)],
            "pc_shipments_yoy": np.r_[np.full(crash_start, 2.0), np.linspace(-5, -15, len(dates) - crash_start)],
            "smartphone_shipments_yoy": np.r_[np.full(crash_start, 1.0), np.linspace(-4, -12, len(dates) - crash_start)],
            "pc_sellthrough_yoy": np.r_[np.linspace(3, -1, crash_start), np.linspace(-2, -16, len(dates) - crash_start)],
            "inventory_days_oem": np.r_[np.linspace(55, 80, crash_start), np.linspace(85, 120, len(dates) - crash_start)],
            "inventory_days_components": np.r_[np.linspace(55, 88, crash_start), np.linspace(95, 140, len(dates) - crash_start)],
            "pcb_revenue_yoy": np.r_[np.linspace(0, 42, crash_start), np.linspace(20, -25, len(dates) - crash_start)],
            "mlcc_revenue_yoy": np.r_[np.linspace(0, 40, crash_start), np.linspace(18, -22, len(dates) - crash_start)],
            "driver_ic_revenue_yoy": np.r_[np.linspace(0, 38, crash_start), np.linspace(15, -20, len(dates) - crash_start)],
            "unit_growth_yoy": np.r_[np.linspace(2, 5, crash_start), np.linspace(2, -10, len(dates) - crash_start)],
            "asp_growth_yoy": np.r_[np.linspace(3, 28, crash_start), np.linspace(15, -5, len(dates) - crash_start)],
            "ai_server_capex_yoy": np.linspace(10, 80, len(dates)),
            "consumer_sentiment": np.r_[np.linspace(100, 88, crash_start), np.linspace(82, 65, len(dates) - crash_start)],
            "cpi_yoy": np.linspace(2.0, 4.5, len(dates)),
            "core_cpi_yoy": np.linspace(2.0, 4.0, len(dates)),
            "ppi_yoy": np.linspace(2.0, 7.0, len(dates)),
            "real_wage_growth_yoy": np.linspace(1.0, -1.0, len(dates)),
            "consumer_confidence": np.r_[np.linspace(100, 86, crash_start), np.linspace(80, 60, len(dates) - crash_start)],
            "unemployment_rate": np.r_[np.linspace(4.0, 4.2, crash_start), np.linspace(4.5, 5.5, len(dates) - crash_start)],
            "policy_rate": np.linspace(2.0, 4.5, len(dates)),
            "us10y_yield": np.linspace(2.5, 4.7, len(dates)),
            "credit_card_delinquency": np.r_[np.linspace(2.0, 2.4, crash_start), np.linspace(2.7, 3.5, len(dates) - crash_start)],
            "oil_price_yoy": np.linspace(0, 25, len(dates)),
            "usd_index": np.linspace(100, 108, len(dates)),
            "retail_sales_yoy": np.linspace(4.0, 0.0, len(dates)),
            "sox_relative_strength": np.r_[np.linspace(10, 20, crash_start), np.linspace(5, -20, len(dates) - crash_start)],
            "tsmc_relative_strength": np.r_[np.linspace(8, 15, crash_start), np.linspace(4, -12, len(dates) - crash_start)],
            "memory_relative_strength": np.r_[np.linspace(5, 25, crash_start), np.linspace(0, -25, len(dates) - crash_start)],
            "pcb_relative_strength": np.r_[np.linspace(4, 18, crash_start), np.linspace(0, -22, len(dates) - crash_start)],
            "mlcc_relative_strength": np.r_[np.linspace(3, 16, crash_start), np.linspace(0, -18, len(dates) - crash_start)],
            "valuation_risk_index": np.linspace(50, 85, len(dates)),
            "liquidity_stress_index": np.r_[np.linspace(30, 50, crash_start), np.linspace(70, 90, len(dates) - crash_start)],
            "event_flag": 0,
        }
    )

    market.to_csv(data_dir / "market.csv", index=False)
    options.to_csv(data_dir / "options.csv", index=False)
    portfolio.to_csv(data_dir / "portfolio.csv", index=False)
    macro.to_csv(data_dir / "macro_factors.csv", index=False)
    return len(market), len(options), len(portfolio)
