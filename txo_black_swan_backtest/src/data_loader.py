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
    "quote_quality_status",
    "spread_pct",
    "is_tradable_quote",
]

PORTFOLIO_COLUMNS = ["date", "stock_equity", "portfolio_beta"]

MACRO_FACTOR_COLUMNS = [
    "date",
    "hbm_asp_index",
    "ddr5_spot_index",
    "pc_shipments_yoy",
    "smartphone_shipments_yoy",
    "pc_sellthrough_yoy",
    "inventory_days_oem",
    "inventory_days_components",
    "pcb_revenue_yoy",
    "mlcc_revenue_yoy",
    "driver_ic_revenue_yoy",
    "unit_growth_yoy",
    "asp_growth_yoy",
    "ai_server_capex_yoy",
    "consumer_sentiment",
    "cpi_yoy",
    "core_cpi_yoy",
    "ppi_yoy",
    "real_wage_growth_yoy",
    "consumer_confidence",
    "unemployment_rate",
    "policy_rate",
    "us10y_yield",
    "credit_card_delinquency",
    "oil_price_yoy",
    "usd_index",
    "retail_sales_yoy",
    "sox_relative_strength",
    "tsmc_relative_strength",
    "memory_relative_strength",
    "pcb_relative_strength",
    "mlcc_relative_strength",
    "valuation_risk_index",
    "liquidity_stress_index",
    "event_flag",
]


def _empty_frame(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def create_sample_csvs(data_dir: Path) -> None:
    """Create empty CSV templates if missing."""

    data_dir.mkdir(parents=True, exist_ok=True)
    for name, columns in {
        "market.csv": MARKET_COLUMNS,
        "options.csv": OPTION_COLUMNS,
        "portfolio.csv": PORTFOLIO_COLUMNS,
        "macro_factors.csv": MACRO_FACTOR_COLUMNS,
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
    bid_missing_initial = "bid" not in options.columns
    ask_missing_initial = "ask" not in options.columns
    if "bid" not in options.columns:
        options["bid"] = np.nan
    if "ask" not in options.columns:
        options["ask"] = np.nan
    options["bid_ask_estimated"] = bid_missing_initial | ask_missing_initial | options["bid"].isna() | options["ask"].isna()
    options["bid"] = options["bid"].fillna(options["close"] * (1.0 - estimated_spread)).clip(lower=0.0)
    options["ask"] = options["ask"].fillna(options["close"] * (1.0 + estimated_spread)).clip(lower=0.0)
    for col in ["volume", "open_interest"]:
        if col not in options.columns:
            options[col] = 0
    if "iv" not in options.columns:
        options["iv"] = np.nan
    if "delta" not in options.columns:
        options["delta"] = np.nan
    quote_quality_available = "is_tradable_quote" in options.columns
    if "quote_quality_status" not in options.columns:
        options["quote_quality_status"] = "UNKNOWN"
    if "spread_pct" not in options.columns:
        options["spread_pct"] = np.nan
    if "is_tradable_quote" not in options.columns:
        options["is_tradable_quote"] = True
    options["is_tradable_quote"] = _bool_series(options["is_tradable_quote"]).fillna(False if quote_quality_available else True)

    underlying = market[["date", "txf_close"]].rename(columns={"txf_close": "underlying"})
    options = options.merge(underlying, on="date", how="left")
    rate = float(config.get("risk_free_rate", 0.015))

    min_volume = float(config.get("wide_spread_volume_threshold", 50))
    min_oi = float(config.get("min_open_interest", 100))
    max_pricing_dte = float(config.get("max_option_pricing_dte", 180))
    needs_greeks = config.get("runtime_mode") != "put_spread_only"
    iv_values: list[float | None] = [None if not valid_number(v) else float(v) for v in options["iv"]]
    delta_values: list[float | None] = [None if not valid_number(v) else float(v) for v in options["delta"]]
    iv_estimated_values: list[bool] = [False] * len(options)
    delta_estimated_values: list[bool] = [False] * len(options)
    tradable: list[bool] = [False] * len(options)
    reasons: list[str] = [""] * len(options)

    basic_mask = (
        options["is_tradable_quote"].astype(bool)
        & options["underlying"].notna()
        & options["dte"].notna()
        & options["bid"].notna()
        & options["ask"].notna()
        & (options["dte"] >= 0)
        & (options["dte"] <= max_pricing_dte)
        & (options["ask"] >= options["bid"])
        & (options["bid"] >= 0)
        & (options["volume"] >= min_volume)
        & (options["open_interest"] >= min_oi)
    )
    reasons = np.where(~options["is_tradable_quote"].astype(bool), "non_tradable_quote", reasons).tolist()
    reasons = np.where((options["volume"] < min_volume) | (options["open_interest"] < min_oi), "insufficient_liquidity", reasons).tolist()
    reasons = np.where((options["dte"].isna()) | (options["dte"] < 0) | (options["dte"] > max_pricing_dte), "invalid_dte", reasons).tolist()
    reasons = np.where((options["bid"].isna()) | (options["ask"].isna()) | (options["ask"] < options["bid"]) | (options["bid"] < 0), "invalid_quote", reasons).tolist()

    if not needs_greeks:
        tradable = basic_mask.tolist()
        reasons = np.where(basic_mask, "", reasons).tolist()
    else:
        for row in options.loc[basic_mask].itertuples():
            quote_ok = True
            iv_input_valid = valid_number(row.iv) and float(row.iv) > 0
            iv = float(row.iv) if iv_input_valid else None
            iv_estimated = False
            if iv is None and valid_number(row.underlying):
                iv = implied_vol(
                    float(row.close),
                    float(row.underlying),
                    float(row.strike),
                    float(row.dte),
                    rate,
                    row.cp,
                )
                iv_estimated = iv is not None
            delta_input_valid = valid_number(row.delta)
            delta = float(row.delta) if delta_input_valid else None
            delta_estimated = False
            if delta is None and iv is not None and valid_number(row.underlying):
                delta = bs_delta(
                    float(row.underlying),
                    float(row.strike),
                    float(row.dte),
                    rate,
                    iv,
                    row.cp,
                )
                delta_estimated = delta == delta
            is_tradable = iv is not None and delta is not None and float(row.ask) >= float(row.bid) >= 0 and quote_ok
            if is_tradable:
                reason = ""
            elif not quote_ok:
                reason = "non_tradable_quote"
            else:
                reason = "missing_iv_or_delta"
            iv_values[row.Index] = iv
            delta_values[row.Index] = delta
            iv_estimated_values[row.Index] = iv_estimated
            delta_estimated_values[row.Index] = delta_estimated
            tradable[row.Index] = is_tradable
            reasons[row.Index] = reason
    options["iv"] = iv_values
    options["delta"] = delta_values
    options["tradable"] = tradable
    options["reason"] = reasons
    options["iv_estimated"] = iv_estimated_values
    options["delta_estimated"] = delta_estimated_values
    return options.sort_values(["date", "expiry", "cp", "strike"]).reset_index(drop=True)


def _bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(float) != 0
    return series.astype(str).str.lower().isin({"true", "1", "yes", "y"})


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


def load_macro_factors(data_dir: Path) -> pd.DataFrame:
    """Load optional macro factors used for supply-chain regime diagnostics."""

    path = data_dir / "macro_factors.csv"
    if not path.exists():
        warn(f"Missing {path}; creating empty template.")
        create_sample_csvs(data_dir)
        return _empty_frame(MACRO_FACTOR_COLUMNS)
    macro = pd.read_csv(path, parse_dates=["date"])
    if macro.empty:
        return macro
    if "date" not in macro.columns:
        raise ValueError("macro_factors.csv missing required column: date")
    for col in MACRO_FACTOR_COLUMNS:
        if col not in macro.columns:
            macro[col] = np.nan if col != "event_flag" else 0
    return macro[MACRO_FACTOR_COLUMNS].sort_values("date").reset_index(drop=True)
