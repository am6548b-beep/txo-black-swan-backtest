"""Adapters from raw TAIFEX-style CSV files to normalized backtest schemas."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data_loader import MACRO_FACTOR_COLUMNS, MARKET_COLUMNS, OPTION_COLUMNS


def load_taifex_futures(raw_dir: Path) -> pd.DataFrame:
    path = _first_existing(raw_dir, ["tx_futures.csv", "taifex_futures.csv", "futures.csv", "market.csv"])
    if path is None:
        return pd.DataFrame(columns=["date", "tx_open", "tx_high", "tx_low", "tx_close", "txf_close", "volume"])
    df = pd.read_csv(path)
    out = pd.DataFrame()
    out["date"] = _date(df, ["date", "trading_date", "Date", "日期", "交易日期"])
    out["tx_open"] = _num(df, ["tx_open", "open", "Open", "開盤價"], fallback=np.nan)
    out["tx_high"] = _num(df, ["tx_high", "high", "High", "最高價"], fallback=np.nan)
    out["tx_low"] = _num(df, ["tx_low", "low", "Low", "最低價"], fallback=np.nan)
    out["tx_close"] = _num(df, ["tx_close", "close", "Close", "收盤價", "結算價"])
    out["txf_close"] = _num(df, ["txf_close", "future_close", "tx_close", "close", "Close", "收盤價", "結算價"])
    out["volume"] = _num(df, ["volume", "Volume", "成交量"], fallback=0)
    for col in ["tx_open", "tx_high", "tx_low"]:
        out[col] = out[col].fillna(out["tx_close"])
    return out.dropna(subset=["date", "tx_close"]).sort_values("date").reset_index(drop=True)


def load_taifex_vix(raw_dir: Path) -> pd.DataFrame:
    path = _first_existing(raw_dir, ["tx_vix.csv", "vix.csv", "taifex_vix.csv"])
    if path is None:
        return pd.DataFrame(columns=["date", "vix"])
    df = pd.read_csv(path)
    return pd.DataFrame(
        {
            "date": _date(df, ["date", "trading_date", "Date", "日期"]),
            "vix": _num(df, ["vix", "VIX", "tx_vix", "指數值", "收盤價"]),
        }
    ).dropna(subset=["date", "vix"]).sort_values("date").reset_index(drop=True)


def load_taifex_options(raw_dir: Path) -> pd.DataFrame:
    path = _first_existing(raw_dir, ["txo_options.csv", "options.csv", "taifex_options.csv"])
    if path is None:
        return pd.DataFrame(columns=OPTION_COLUMNS)
    df = pd.read_csv(path)
    out = pd.DataFrame()
    out["date"] = _date(df, ["date", "trading_date", "Date", "日期", "交易日期"])
    out["expiry"] = _date(df, ["expiry", "expiration", "到期日", "契約月份"])
    out["cp"] = _cp(df, ["cp", "call_put", "option_type", "買賣權", "買賣權別"])
    out["strike"] = _num(df, ["strike", "履約價", "履約價格"])
    out["close"] = _num(df, ["close", "Close", "收盤價", "結算價"])
    out["bid"] = _num(df, ["bid", "best_bid", "買價", "最佳買價"], fallback=np.nan)
    out["ask"] = _num(df, ["ask", "best_ask", "賣價", "最佳賣價"], fallback=np.nan)
    out["volume"] = _num(df, ["volume", "Volume", "成交量"], fallback=0)
    out["open_interest"] = _num(df, ["open_interest", "oi", "未沖銷契約數", "未平倉量"], fallback=0)
    out["iv"] = _num(df, ["iv", "IV", "implied_vol", "隱含波動率"], fallback=np.nan)
    out["delta"] = _num(df, ["delta", "Delta"], fallback=np.nan)
    out["dte"] = (pd.to_datetime(out["expiry"]) - pd.to_datetime(out["date"])).dt.days
    out["quote_quality_status"] = "UNKNOWN"
    out["spread_pct"] = np.nan
    out["is_tradable_quote"] = True
    out = out.dropna(subset=["date", "expiry", "cp", "strike", "close"])
    return out[OPTION_COLUMNS].sort_values(["date", "expiry", "cp", "strike"]).reset_index(drop=True)


def load_macro_factors_raw(raw_dir: Path) -> pd.DataFrame:
    path = _first_existing(raw_dir, ["macro_factors.csv", "macro.csv"])
    if path is None:
        return pd.DataFrame(columns=MACRO_FACTOR_COLUMNS)
    df = pd.read_csv(path)
    out = pd.DataFrame()
    out["date"] = _date(df, ["date", "Date", "日期"])
    for col in MACRO_FACTOR_COLUMNS:
        if col == "date":
            continue
        out[col] = _num(df, [col], fallback=0 if col == "event_flag" else np.nan)
    return out[MACRO_FACTOR_COLUMNS].dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def build_market_frame(futures: pd.DataFrame, vix: pd.DataFrame) -> pd.DataFrame:
    if futures.empty:
        return pd.DataFrame(columns=MARKET_COLUMNS)
    market = futures.copy()
    if not vix.empty:
        market = market.merge(vix, on="date", how="left")
    if "vix" not in market:
        market["vix"] = np.nan
    market["event_flag"] = 0
    for col in MARKET_COLUMNS:
        if col not in market:
            market[col] = np.nan
    return market[MARKET_COLUMNS].sort_values("date").reset_index(drop=True)


def _first_existing(raw_dir: Path, names: list[str]) -> Path | None:
    for name in names:
        path = raw_dir / name
        if path.exists():
            return path
    for name in names:
        matches = sorted(p for p in raw_dir.rglob(name) if p.is_file())
        if matches:
            return matches[0]
    return None


def _date(df: pd.DataFrame, aliases: list[str]) -> pd.Series:
    col = _find_col(df, aliases)
    if col is None:
        return pd.Series(pd.NaT, index=df.index)
    return pd.to_datetime(df[col], errors="coerce")


def _num(df: pd.DataFrame, aliases: list[str], fallback: float | None = None) -> pd.Series:
    col = _find_col(df, aliases)
    if col is None:
        return pd.Series(fallback, index=df.index, dtype="float64")
    values = df[col].astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False)
    return pd.to_numeric(values, errors="coerce")


def _cp(df: pd.DataFrame, aliases: list[str]) -> pd.Series:
    col = _find_col(df, aliases)
    if col is None:
        return pd.Series(np.nan, index=df.index)
    values = df[col].astype(str).str.upper().str.strip()
    return values.replace({"CALL": "C", "PUT": "P", "買權": "C", "賣權": "P"})


def _find_col(df: pd.DataFrame, aliases: list[str]) -> str | None:
    exact = {str(c): c for c in df.columns}
    lower = {str(c).lower(): c for c in df.columns}
    for alias in aliases:
        if alias in exact:
            return exact[alias]
        if alias.lower() in lower:
            return lower[alias.lower()]
    return None
