"""Data quality checks for market, option, and portfolio CSV files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .data_loader import MACRO_FACTOR_COLUMNS, MARKET_COLUMNS, OPTION_COLUMNS, PORTFOLIO_COLUMNS


def validate_data(data_dir: Path, report_dir: Path, config: dict) -> pd.DataFrame:
    """Validate input CSV shape and conservative execution prerequisites."""

    rows: list[dict] = []
    rows.extend(_validate_csv(data_dir / "market.csv", MARKET_COLUMNS, required=["date", "tx_close"]))
    rows.extend(
        _validate_csv(
            data_dir / "options.csv",
            OPTION_COLUMNS,
            required=["date", "expiry", "dte", "cp", "strike", "close"],
        )
    )
    rows.extend(_validate_csv(data_dir / "portfolio.csv", PORTFOLIO_COLUMNS, required=[]))
    rows.extend(_validate_csv(data_dir / "macro_factors.csv", MACRO_FACTOR_COLUMNS, required=["date"]))

    market = _read_csv(data_dir / "market.csv", ["date"])
    options = _read_csv(data_dir / "options.csv", ["date", "expiry"])
    portfolio = _read_csv(data_dir / "portfolio.csv", ["date"])
    macro = _read_csv(data_dir / "macro_factors.csv", ["date"])

    if not market.empty:
        rows.extend(_date_checks("market.csv", market, "date"))
        vix_ok = "vix" in market and market["vix"].notna().any()
        rows.append(_check("market.csv", "vix_available", "PASS" if vix_ok else "WARN", "vix present" if vix_ok else "missing vix will use realized-vol proxy"))
        event_ok = "event_flag" in market
        rows.append(_check("market.csv", "event_flag_available", "PASS" if event_ok else "WARN", "event_flag present" if event_ok else "missing event_flag defaults to 0"))
    if not options.empty:
        rows.extend(_date_checks("options.csv", options, "date"))
        bidask_ok = {"bid", "ask"} <= set(options.columns) and options[["bid", "ask"]].notna().all().all()
        rows.append(_check("options.csv", "bid_ask_available", "PASS" if bidask_ok else "WARN", "bid/ask present" if bidask_ok else "missing bid/ask will be estimated and may be optimistic"))
        rows.append(_check("options.csv", "bid_ask_valid", "PASS" if {"bid", "ask"} <= set(options.columns) and (options["ask"] >= options["bid"]).all() else "FAIL", "ask must be >= bid"))
        iv_ok = "iv" in options and options["iv"].notna().any()
        delta_ok = "delta" in options and options["delta"].notna().any()
        rows.append(_check("options.csv", "iv_available", "PASS" if iv_ok else "WARN", "iv present" if iv_ok else "missing iv requires inversion; failures are untradable"))
        rows.append(_check("options.csv", "delta_available", "PASS" if delta_ok else "WARN", "delta present" if delta_ok else "missing delta uses Black-Scholes approximation"))
        rows.append(_check("options.csv", "liquidity_columns", "PASS" if {"volume", "open_interest"} <= set(options.columns) else "FAIL", "volume/open_interest required for liquidity gating"))
        if {"volume", "open_interest"} <= set(options.columns):
            liquid = (
                (options["volume"] >= float(config.get("wide_spread_volume_threshold", 50)))
                & (options["open_interest"] >= float(config.get("min_open_interest", 100)))
            )
            rows.append(_check("options.csv", "liquid_contract_rows", "PASS" if liquid.any() else "WARN", f"liquid rows={int(liquid.sum())}"))
    if not portfolio.empty:
        rows.extend(_date_checks("portfolio.csv", portfolio, "date"))
        beta_ok = "portfolio_beta" in portfolio
        rows.append(_check("portfolio.csv", "portfolio_beta_available", "PASS" if beta_ok else "WARN", "portfolio_beta present" if beta_ok else "missing portfolio uses config beta"))
    if not macro.empty:
        rows.extend(_date_checks("macro_factors.csv", macro, "date"))
        supply_cols = {"hbm_asp_index", "ddr5_spot_index", "inventory_days_components", "pc_sellthrough_yoy"}
        fragility_cols = {"cpi_yoy", "core_cpi_yoy", "real_wage_growth_yoy", "consumer_confidence"}
        rows.append(_check("macro_factors.csv", "supply_distortion_columns", "PASS" if supply_cols <= set(macro.columns) else "WARN", "AI supply distortion inputs present" if supply_cols <= set(macro.columns) else "missing some AI supply inputs"))
        rows.append(_check("macro_factors.csv", "macro_fragility_columns", "PASS" if fragility_cols <= set(macro.columns) else "WARN", "macro demand fragility inputs present" if fragility_cols <= set(macro.columns) else "missing some macro fragility inputs"))

    report = pd.DataFrame(rows)
    report_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(report_dir / "data_quality_report.csv", index=False)
    return report


def _validate_csv(path: Path, expected_columns: list[str], required: list[str]) -> list[dict]:
    if not path.exists():
        return [_check(path.name, "file_exists", "FAIL", "file missing")]
    try:
        df = pd.read_csv(path, nrows=5)
    except Exception as exc:
        return [_check(path.name, "readable", "FAIL", str(exc))]
    rows = [_check(path.name, "file_exists", "PASS", "")]
    missing_required = [c for c in required if c not in df.columns]
    missing_expected = [c for c in expected_columns if c not in df.columns]
    rows.append(_check(path.name, "required_columns", "PASS" if not missing_required else "FAIL", ",".join(missing_required)))
    rows.append(_check(path.name, "expected_columns", "PASS" if not missing_expected else "WARN", ",".join(missing_expected)))
    row_count = len(pd.read_csv(path))
    rows.append(_check(path.name, "has_rows", "PASS" if row_count > 0 else "WARN", f"rows={row_count}" if row_count > 0 else "empty file"))
    return rows


def _read_csv(path: Path, date_cols: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, parse_dates=[c for c in date_cols if c in pd.read_csv(path, nrows=0).columns])
    except Exception:
        return pd.DataFrame()


def _date_checks(file_name: str, df: pd.DataFrame, date_col: str) -> list[dict]:
    if date_col not in df.columns:
        return [_check(file_name, "date_column", "FAIL", f"missing {date_col}")]
    return [
        _check(file_name, "date_not_null", "PASS" if df[date_col].notna().all() else "FAIL", ""),
        _check(
            file_name,
            "date_unique_or_chain",
            "PASS" if file_name == "options.csv" or not df[date_col].duplicated().any() else "WARN",
            "option chain allows duplicate dates" if file_name == "options.csv" else ("duplicate dates" if df[date_col].duplicated().any() else ""),
        ),
    ]


def _check(file_name: str, check: str, status: str, detail: str) -> dict:
    return {"file": file_name, "check": check, "status": status, "detail": detail}
