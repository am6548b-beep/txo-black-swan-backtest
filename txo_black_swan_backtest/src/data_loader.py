"""CSV loading and conservative data normalization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .option_pricing import bs_delta, implied_vol, valid_number
from .utils import warn


MARKET_COLUMNS = [
    "date",
    "tx_close",
    "tx_open",
    "tx_high",
    "tx_low",
    "txf_close",
    "volume",
    "vix",
    "event_flag",
]

OPTION_COLUMNS = [
    "date",
    "expiry",
    "dte",
    "cp",
    "strike",
    "close",
    "bid",
    "ask",
    "volume",
    "open_interest",
    "iv",
    "delta",
]

PORTFOLIO_COLUMNS = ["date", "stock_equity", "portfolio_beta"]


def _empty_frame(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def create_sample_csvs(data_dir: Path) -> None:
    """Create empty CSV templates if missing."""

    data_dir.mkdir(parents=True, exist_ok=True)
    for name, columns in {
        "market.csv": MARKET_COLUMNS,
        "options.csv": OPTION_COLUMNS,
        "portfolio.csv": PORTFOLIO_COLUMNS,
    }.items():
        path = data_dir / name
        if not path.exists():
            pd.DataFrame(columns=columns).to_csv(path, index=False)


def load_market(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "market.csv"
    if not path.exists():
        warn(f"Missing {path}; creating empty template.")
        create_sample_csvs(data_dir)
        return _empty_frame(MARKET_COLUMNS)
    market = pd.read_csv(path, parse_dates=["date"])
    if market.empty:
        return market
    required = {"date", "tx_close"}
    missing = required - set(market.columns)
    if missing:
        raise ValueError(f"market.csv missing required columns: {sorted(missing)}")
    for col in ["tx_open", "tx_high", "tx_low", "txf_close"]:
        if col not in market.columns:
            market[col] = market["tx_close"]
    if "volume" not in market.columns:
        market["volume"] = 0
    if "event_flag" not in market.columns:
        market["event_flag"] = 0
    market["vix_is_proxy"] = False
    if "vix" not in market.columns or market["vix"].isna().all():
        ret = market["tx_close"].pct_change()
        market["vix"] = ret.rolling(20, min_periods=20).std() * np.sqrt(252) * 100
        market["vix_is_proxy"] = True
        warn("market.csv lacks vix; using 20-day realized volatility proxy.")
    market = market.sort_values("date").reset_index(drop=True)
    return market


def load_options(data_dir: Path, market: pd.DataFrame, config: dict) -> pd.DataFrame:
    path = data_dir / "options.csv"
    if not path.exists():
        warn(f"Missing {path}; creating empty template.")
        create_sample_csvs(data_dir)
        return _empty_frame(OPTION_COLUMNS)
    options = pd.read_csv(path, parse_dates=["date", "expiry"])
    if options.empty:
        return options
    missing = {"date", "expiry", "dte", "cp", "strike", "close"} - set(options.columns)
    if missing:
        raise ValueError(f"options.csv missing required columns: {sorted(missing)}")

    options["cp"] = options["cp"].str.upper()
    estimated_spread = float(config.get("estimated_spread", 0.08))
    if "bid" not in options.columns:
        options["bid"] = np.nan
    if "ask" not in options.columns:
        options["ask"] = np.nan
    options["bid"] = options["bid"].fillna(options["close"] * (1.0 - estimated_spread)).clip(lower=0.0)
    options["ask"] = options["ask"].fillna(options["close"] * (1.0 + estimated_spread)).clip(lower=0.0)
    for col in ["volume", "open_interest"]:
        if col not in options.columns:
            options[col] = 0
    if "iv" not in options.columns:
        options["iv"] = np.nan
    if "delta" not in options.columns:
        options["delta"] = np.nan

    underlying = market[["date", "txf_close"]].rename(columns={"txf_close": "underlying"})
    options = options.merge(underlying, on="date", how="left")
    rate = float(config.get("risk_free_rate", 0.015))

    iv_values: list[float | None] = []
    delta_values: list[float | None] = []
    tradable: list[bool] = []
    reasons: list[str] = []
    for row in options.itertuples(index=False):
        iv = float(row.iv) if valid_number(row.iv) and float(row.iv) > 0 else None
        if iv is None and valid_number(row.underlying):
            iv = implied_vol(
                float(row.close),
                float(row.underlying),
                float(row.strike),
                float(row.dte),
                rate,
                row.cp,
            )
        delta = float(row.delta) if valid_number(row.delta) else None
        if delta is None and iv is not None and valid_number(row.underlying):
            delta = bs_delta(
                float(row.underlying),
                float(row.strike),
                float(row.dte),
                rate,
                iv,
                row.cp,
            )
        is_tradable = iv is not None and delta is not None and float(row.ask) >= float(row.bid) >= 0
        reason = "" if is_tradable else "missing_iv_or_delta"
        iv_values.append(iv)
        delta_values.append(delta)
        tradable.append(is_tradable)
        reasons.append(reason)
    options["iv"] = iv_values
    options["delta"] = delta_values
    options["tradable"] = tradable
    options["reason"] = reasons
    return options.sort_values(["date", "expiry", "cp", "strike"]).reset_index(drop=True)


def load_portfolio(data_dir: Path, market: pd.DataFrame, config: dict) -> pd.DataFrame:
    path = data_dir / "portfolio.csv"
    if not path.exists() or path.stat().st_size == 0:
        return _synthetic_portfolio(market, config)
    portfolio = pd.read_csv(path, parse_dates=["date"])
    if portfolio.empty:
        return _synthetic_portfolio(market, config)
    missing = {"date", "stock_equity", "portfolio_beta"} - set(portfolio.columns)
    if missing:
        raise ValueError(f"portfolio.csv missing required columns: {sorted(missing)}")
    return portfolio.sort_values("date").reset_index(drop=True)


def _synthetic_portfolio(market: pd.DataFrame, config: dict) -> pd.DataFrame:
    if market.empty:
        return _empty_frame(PORTFOLIO_COLUMNS)
    beta = float(config.get("portfolio_beta", 1.3))
    equity = float(config.get("initial_stock_equity", 1_200_000.0))
    returns = market["tx_close"].pct_change().fillna(0.0)
    stock_equity = [equity]
    for r in returns.iloc[1:]:
        stock_equity.append(stock_equity[-1] * (1.0 + beta * float(r)))
    return pd.DataFrame(
        {
            "date": market["date"],
            "stock_equity": stock_equity,
            "portfolio_beta": beta,
        }
    )

