"""Build local TXO-chain derived risk indicators.

These indicators are diagnostics/proxies derived from local option quotes. They
are not official VIX and must not be treated as high-confidence official data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


LOCAL_RISK_COLUMNS = [
    "date",
    "put_volume",
    "call_volume",
    "put_call_volume_ratio_all",
    "put_oi",
    "call_oi",
    "put_call_oi_ratio_all",
    "put_call_volume_ratio_tradable",
    "put_call_oi_ratio_tradable",
    "atm_put_premium_ratio_30d",
    "atm_call_premium_ratio_30d",
    "atm_straddle_premium_ratio_30d",
    "put_skew_proxy_30d",
    "iv_term_structure_proxy",
    "put_call_source",
    "atm_iv_proxy_source",
    "put_skew_source",
    "term_structure_source",
    "iv_proxy_source",
    "tw_vix_is_proxy",
    "source_coverage_score",
]


def build_local_risk_indicators(processed_dir: Path, report_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    processed_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    market_path = processed_dir / "market.csv"
    options_path = processed_dir / "options.csv"
    if not market_path.exists() or not options_path.exists():
        out = pd.DataFrame(columns=LOCAL_RISK_COLUMNS)
        audit = pd.DataFrame([{"check": "input_files", "status": "FAIL", "detail": "market.csv or options.csv missing"}])
        out.to_csv(processed_dir / "risk_indicators.csv", index=False)
        audit.to_csv(report_dir / "local_risk_indicator_build_audit.csv", index=False)
        (report_dir / "local_risk_indicator_build_audit.md").write_text(_audit_markdown(audit), encoding="utf-8")
        return out, audit

    market = pd.read_csv(market_path, parse_dates=["date"])
    options = pd.read_csv(options_path, parse_dates=["date", "expiry"], low_memory=False)
    indicators = build_local_risk_indicator_frame(market, options)
    audit = local_risk_indicator_audit(indicators, options)
    missing_root_cause = local_txo_proxy_missing_root_cause(market, options)
    indicators.to_csv(processed_dir / "risk_indicators.csv", index=False)
    audit.to_csv(report_dir / "local_risk_indicator_build_audit.csv", index=False)
    missing_root_cause.to_csv(report_dir / "local_txo_proxy_missing_root_cause.csv", index=False)
    (report_dir / "local_risk_indicator_build_audit.md").write_text(_audit_markdown(audit), encoding="utf-8")
    (report_dir / "local_txo_proxy_missing_root_cause.md").write_text(_root_cause_markdown(missing_root_cause), encoding="utf-8")
    return indicators, audit


def build_local_risk_indicator_frame(market: pd.DataFrame, options: pd.DataFrame) -> pd.DataFrame:
    market = market.copy()
    options = options.copy()
    market["date"] = pd.to_datetime(market["date"], errors="coerce")
    options["date"] = pd.to_datetime(options["date"], errors="coerce")
    options["expiry"] = pd.to_datetime(options["expiry"], errors="coerce")
    _ensure_quality_columns(options)
    options["mid"] = (pd.to_numeric(options["bid"], errors="coerce") + pd.to_numeric(options["ask"], errors="coerce")) / 2.0
    options["volume"] = pd.to_numeric(options["volume"], errors="coerce")
    options["open_interest"] = pd.to_numeric(options["open_interest"], errors="coerce")
    options["strike"] = pd.to_numeric(options["strike"], errors="coerce")
    options["dte"] = pd.to_numeric(options["dte"], errors="coerce")

    rows: list[dict[str, Any]] = []
    market_by_date = market.set_index("date")
    for date, group in options.groupby("date", sort=True):
        if pd.isna(date):
            continue
        txf_close = _market_txf_close(market_by_date, date)
        row = {"date": date}
        row.update(_put_call_ratios(group))
        row.update(_premium_proxies(group, txf_close))
        row["put_call_source"] = "local_txo_chain" if pd.notna(row.get("put_call_volume_ratio_all")) else ""
        row["atm_iv_proxy_source"] = "local_txo_chain_mid_quote" if pd.notna(row.get("atm_straddle_premium_ratio_30d")) else ""
        row["put_skew_source"] = "local_txo_chain_mid_quote" if pd.notna(row.get("put_skew_proxy_30d")) else ""
        row["term_structure_source"] = "local_txo_chain_mid_quote" if pd.notna(row.get("iv_term_structure_proxy")) else ""
        row["iv_proxy_source"] = "local_txo_chain_proxy" if any(pd.notna(row.get(col)) for col in ["atm_straddle_premium_ratio_30d", "put_skew_proxy_30d", "iv_term_structure_proxy"]) else ""
        row["tw_vix_is_proxy"] = True
        rows.append(row)
    out = pd.DataFrame(rows)
    for col in LOCAL_RISK_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    value_cols = [
        "put_call_volume_ratio_tradable",
        "put_call_oi_ratio_tradable",
        "atm_straddle_premium_ratio_30d",
        "put_skew_proxy_30d",
        "iv_term_structure_proxy",
    ]
    out["source_coverage_score"] = out[value_cols].notna().mean(axis=1)
    return out[LOCAL_RISK_COLUMNS].sort_values("date")


def local_risk_indicator_audit(indicators: pd.DataFrame, options: pd.DataFrame) -> pd.DataFrame:
    _ensure_quality_columns(options)
    rows = [
        {"check": "options_rows_used", "status": "PASS", "value": int(len(options))},
        {
            "check": "tradable_option_ratio",
            "status": "PASS",
            "value": _safe_ratio(options["is_tradable_quote"].astype(bool).sum(), len(options)),
        },
        {"check": "dates_with_put_call_ratio", "status": "PASS", "value": int(indicators["put_call_volume_ratio_all"].notna().sum())},
        {"check": "dates_with_atm_straddle_proxy", "status": "PASS", "value": int(indicators["atm_straddle_premium_ratio_30d"].notna().sum())},
        {"check": "dates_with_put_skew_proxy", "status": "PASS", "value": int(indicators["put_skew_proxy_30d"].notna().sum())},
        {"check": "dates_with_term_structure_proxy", "status": "PASS", "value": int(indicators["iv_term_structure_proxy"].notna().sum())},
        {"check": "no_lookahead_confirmation", "status": "PASS", "detail": "each row uses same-date market and option chain only"},
    ]
    for col in indicators.columns:
        if col != "date":
            rows.append({"check": "missing_ratio", "field": col, "status": "WARN" if indicators[col].isna().any() else "PASS", "value": float(indicators[col].isna().mean())})
    for status, count in options["quote_quality_status"].fillna("UNKNOWN").astype(str).value_counts().items():
        rows.append({"check": "quote_quality_distribution", "quote_quality_status": status, "status": "PASS", "value": int(count)})
    return pd.DataFrame(rows)


def local_txo_proxy_missing_root_cause(market: pd.DataFrame, options: pd.DataFrame) -> pd.DataFrame:
    """Diagnose why local TXO volatility proxies are unavailable around 2020."""

    market = market.copy()
    options = options.copy()
    market["date"] = pd.to_datetime(market["date"], errors="coerce")
    options["date"] = pd.to_datetime(options["date"], errors="coerce")
    if "expiry" in options.columns:
        options["expiry"] = pd.to_datetime(options["expiry"], errors="coerce")
    _prepare_option_numeric_columns(options)
    _ensure_quality_columns(options)
    options["mid"] = (pd.to_numeric(options.get("bid", np.nan), errors="coerce") + pd.to_numeric(options.get("ask", np.nan), errors="coerce")) / 2.0

    start = pd.Timestamp("2020-01-01") - pd.Timedelta(days=180)
    end = pd.Timestamp("2020-12-31")
    market_dates = set(market["date"].dropna())
    option_dates = set(options["date"].dropna())
    all_dates = sorted(date for date in market_dates.union(option_dates) if start <= date <= end)
    market_by_date = market.set_index("date") if "date" in market else pd.DataFrame()
    options_by_date = {pd.Timestamp(date): group.copy() for date, group in options.groupby("date", sort=False)}
    daily_rows = []
    for date in all_dates:
        market_row_exists = date in market_dates
        txf_close = _market_txf_close(market_by_date, date) if market_row_exists else float("nan")
        group = options_by_date.get(date, pd.DataFrame(columns=options.columns))
        daily_rows.append(_root_cause_daily_row(date, market_row_exists, txf_close, group))
    rows: list[dict[str, Any]] = daily_rows
    rows.extend(_root_cause_aggregate_rows(pd.DataFrame(daily_rows), market, options))
    return pd.DataFrame(rows)


def classify_local_proxy_root_cause(row: dict[str, Any]) -> str:
    """Classify the first blocking reason for same-day local proxy construction."""

    if not bool(row.get("market_row_exists", False)):
        return "NO_MARKET_ROW"
    if not bool(row.get("txf_close_valid", False)):
        return "NO_MARKET_ROW"
    if int(row.get("options_rows_count", 0) or 0) == 0:
        return "NO_OPTIONS_ROWS"
    if int(row.get("tradable_options_count", 0) or 0) == 0:
        return "NO_TRADABLE_OPTIONS"
    if int(row.get("dte_20_45_count", 0) or 0) == 0:
        return "NO_DTE_20_45"
    if int(row.get("dte_20_45_tradable_count", 0) or 0) == 0:
        return "NO_TRADABLE_DTE_20_45"
    if not bool(row.get("atm_call_candidate_exists", False)):
        return "NO_ATM_CALL"
    if not bool(row.get("atm_put_candidate_exists", False)):
        return "NO_ATM_PUT"
    if not bool(row.get("atm_call_valid", False)):
        return "NON_VALID_ATM_CALL"
    if not bool(row.get("atm_put_valid", False)):
        return "NON_VALID_ATM_PUT"
    if not bool(row.get("otm_put_90_candidate_exists", False)):
        return "NO_OTM_PUT_90"
    if not bool(row.get("otm_put_90_valid", False)):
        return "NON_VALID_OTM_PUT_90"
    if not bool(row.get("far_atm_straddle_exists", False)):
        return "NO_FAR_TERM_STRUCTURE"
    return "UNKNOWN"


def _root_cause_daily_row(date: pd.Timestamp, market_row_exists: bool, txf_close: float, group: pd.DataFrame) -> dict[str, Any]:
    txf_close_valid = bool(np.isfinite(txf_close) and txf_close > 0)
    tradable = group[_tradable_mask(group)].copy() if not group.empty else pd.DataFrame(columns=group.columns)
    dte = pd.to_numeric(group.get("dte", pd.Series(dtype=float)), errors="coerce")
    near_all = group[dte.between(20, 45)].copy() if not group.empty else pd.DataFrame(columns=group.columns)
    far_all = group[dte.between(60, 120)].copy() if not group.empty else pd.DataFrame(columns=group.columns)
    near_tradable = tradable[pd.to_numeric(tradable.get("dte", pd.Series(dtype=float)), errors="coerce").between(20, 45)].copy() if not tradable.empty else pd.DataFrame(columns=group.columns)
    far_tradable = tradable[pd.to_numeric(tradable.get("dte", pd.Series(dtype=float)), errors="coerce").between(60, 120)].copy() if not tradable.empty else pd.DataFrame(columns=group.columns)
    atm_call = _nearest_contract(near_all, "C", txf_close) if txf_close_valid else None
    atm_put = _nearest_contract(near_all, "P", txf_close) if txf_close_valid else None
    otm_put = _nearest_contract(near_all, "P", txf_close * 0.90) if txf_close_valid else None
    near_put_valid = _nearest_contract(near_tradable, "P", txf_close) if txf_close_valid else None
    near_call_valid = _nearest_contract(near_tradable, "C", txf_close) if txf_close_valid else None
    far_put_valid = _nearest_contract(far_tradable, "P", txf_close) if txf_close_valid else None
    far_call_valid = _nearest_contract(far_tradable, "C", txf_close) if txf_close_valid else None
    row: dict[str, Any] = {
        "section": "daily_root_cause",
        "date": date,
        "analysis_window": "pre_crash_180d" if date < pd.Timestamp("2020-01-01") else "full_year_2020",
        "market_row_exists": bool(market_row_exists),
        "txf_close": txf_close,
        "txf_close_valid": txf_close_valid,
        "options_rows_count": int(len(group)),
        "tradable_options_count": int(len(tradable)),
        "valid_quote_count": int(group.get("quote_quality_status", pd.Series(dtype=str)).fillna("").astype(str).eq("VALID").sum()) if not group.empty else 0,
        "dte_20_45_count": int(len(near_all)),
        "dte_20_45_tradable_count": int(len(near_tradable)),
        "dte_60_120_count": int(len(far_all)),
        "dte_60_120_tradable_count": int(len(far_tradable)),
        "atm_call_candidate_exists": atm_call is not None,
        "atm_put_candidate_exists": atm_put is not None,
        "atm_call_valid": _contract_is_valid(atm_call),
        "atm_put_valid": _contract_is_valid(atm_put),
        "atm_call_strike": _contract_value(atm_call, "strike"),
        "atm_put_strike": _contract_value(atm_put, "strike"),
        "atm_call_dte": _contract_value(atm_call, "dte"),
        "atm_put_dte": _contract_value(atm_put, "dte"),
        "atm_call_quote_quality_status": _contract_value(atm_call, "quote_quality_status", ""),
        "atm_put_quote_quality_status": _contract_value(atm_put, "quote_quality_status", ""),
        "atm_call_spread_pct": _contract_value(atm_call, "spread_pct"),
        "atm_put_spread_pct": _contract_value(atm_put, "spread_pct"),
        "otm_put_90_candidate_exists": otm_put is not None,
        "otm_put_90_valid": _contract_is_valid(otm_put),
        "otm_put_90_strike": _contract_value(otm_put, "strike"),
        "otm_put_90_dte": _contract_value(otm_put, "dte"),
        "otm_put_90_quote_quality_status": _contract_value(otm_put, "quote_quality_status", ""),
        "near_atm_straddle_exists": near_put_valid is not None and near_call_valid is not None,
        "far_atm_straddle_exists": far_put_valid is not None and far_call_valid is not None,
        "near_atm_straddle_valid": near_put_valid is not None and near_call_valid is not None,
        "far_atm_straddle_valid": far_put_valid is not None and far_call_valid is not None,
    }
    row["root_cause"] = classify_local_proxy_root_cause(row)
    return row


def _root_cause_aggregate_rows(daily: pd.DataFrame, market: pd.DataFrame, options: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    daily_rows = daily[daily.get("section", pd.Series(dtype=str)) == "daily_root_cause"].copy()
    if not daily_rows.empty:
        for cause, count in daily_rows["root_cause"].fillna("UNKNOWN").astype(str).value_counts().items():
            rows.append({"section": "aggregate_summary", "metric": "root_cause_distribution", "root_cause": cause, "days": int(count)})
    opt2020 = options[(pd.to_datetime(options["date"], errors="coerce") >= pd.Timestamp("2020-01-01")) & (pd.to_datetime(options["date"], errors="coerce") <= pd.Timestamp("2020-12-31"))].copy()
    if not opt2020.empty:
        opt2020["month"] = pd.to_datetime(opt2020["date"], errors="coerce").dt.to_period("M").astype(str)
        for month, group in opt2020.groupby("month", sort=True):
            rows.append(
                {
                    "section": "aggregate_summary",
                    "metric": "options_rows_by_month_2020",
                    "month": month,
                    "options_rows": int(len(group)),
                    "tradable_options_rows": int(_tradable_mask(group).sum()),
                }
            )
        for status, count in opt2020["quote_quality_status"].fillna("UNKNOWN").astype(str).value_counts().items():
            rows.append({"section": "aggregate_summary", "metric": "quote_quality_status_distribution_2020", "quote_quality_status": status, "rows": int(count)})
        buckets = pd.cut(pd.to_numeric(opt2020.get("dte", np.nan), errors="coerce"), bins=[-np.inf, 0, 19, 45, 120, np.inf], labels=["lt0", "1_19", "20_45", "46_120", "gt120"])
        for bucket, count in buckets.value_counts(dropna=False).items():
            rows.append({"section": "aggregate_summary", "metric": "dte_bucket_distribution_2020", "dte_bucket": str(bucket), "rows": int(count)})
    market_dates = set(pd.to_datetime(market.get("date", pd.Series(dtype="datetime64[ns]")), errors="coerce").dropna())
    option_dates = set(pd.to_datetime(options.get("date", pd.Series(dtype="datetime64[ns]")), errors="coerce").dropna())
    dates2020 = {date for date in market_dates.union(option_dates) if pd.Timestamp("2020-01-01") <= date <= pd.Timestamp("2020-12-31")}
    aligned = len([date for date in dates2020 if date in market_dates and date in option_dates])
    rows.append(
        {
            "section": "aggregate_summary",
            "metric": "date_alignment_2020",
            "market_dates": int(len([date for date in market_dates if pd.Timestamp("2020-01-01") <= date <= pd.Timestamp("2020-12-31")])),
            "options_dates": int(len([date for date in option_dates if pd.Timestamp("2020-01-01") <= date <= pd.Timestamp("2020-12-31")])),
            "aligned_dates": int(aligned),
            "market_options_dates_align": bool(aligned > 0 and aligned == len(dates2020)),
        }
    )
    rows.append(
        {
            "section": "aggregate_summary",
            "metric": "processed_options_2020_coverage",
            "options_rows_2020": int(len(opt2020)),
            "has_processed_options_2020": bool(len(opt2020) > 0),
        }
    )
    return rows


def _put_call_ratios(group: pd.DataFrame) -> dict[str, float]:
    puts = group[group["cp"].astype(str).str.upper() == "P"]
    calls = group[group["cp"].astype(str).str.upper() == "C"]
    tradable = group[_tradable_mask(group)]
    tputs = tradable[tradable["cp"].astype(str).str.upper() == "P"]
    tcalls = tradable[tradable["cp"].astype(str).str.upper() == "C"]
    put_volume = puts["volume"].sum(min_count=1)
    call_volume = calls["volume"].sum(min_count=1)
    put_oi = puts["open_interest"].sum(min_count=1)
    call_oi = calls["open_interest"].sum(min_count=1)
    return {
        "put_volume": put_volume,
        "call_volume": call_volume,
        "put_call_volume_ratio_all": _safe_ratio(put_volume, call_volume),
        "put_oi": put_oi,
        "call_oi": call_oi,
        "put_call_oi_ratio_all": _safe_ratio(put_oi, call_oi),
        "put_call_volume_ratio_tradable": _safe_ratio(tputs["volume"].sum(min_count=1), tcalls["volume"].sum(min_count=1)),
        "put_call_oi_ratio_tradable": _safe_ratio(tputs["open_interest"].sum(min_count=1), tcalls["open_interest"].sum(min_count=1)),
    }


def _prepare_option_numeric_columns(options: pd.DataFrame) -> None:
    for col in ["bid", "ask", "volume", "open_interest", "strike", "dte", "spread_pct"]:
        if col not in options.columns:
            options[col] = np.nan
        options[col] = pd.to_numeric(options[col], errors="coerce")
    if "cp" not in options.columns:
        options["cp"] = ""


def _premium_proxies(group: pd.DataFrame, txf_close: float) -> dict[str, float]:
    out = {
        "atm_put_premium_ratio_30d": np.nan,
        "atm_call_premium_ratio_30d": np.nan,
        "atm_straddle_premium_ratio_30d": np.nan,
        "put_skew_proxy_30d": np.nan,
        "iv_term_structure_proxy": np.nan,
    }
    if not np.isfinite(txf_close) or txf_close <= 0:
        return out
    tradable = group[_tradable_mask(group)].copy()
    near = tradable[tradable["dte"].between(20, 45)].copy()
    far = tradable[tradable["dte"].between(60, 120)].copy()
    atm_put = _nearest_contract(near, "P", txf_close)
    atm_call = _nearest_contract(near, "C", txf_close)
    if atm_put is not None:
        out["atm_put_premium_ratio_30d"] = float(atm_put["mid"] / txf_close)
    if atm_call is not None:
        out["atm_call_premium_ratio_30d"] = float(atm_call["mid"] / txf_close)
    if atm_put is not None and atm_call is not None:
        out["atm_straddle_premium_ratio_30d"] = float((atm_put["mid"] + atm_call["mid"]) / txf_close)
    otm_put = _nearest_contract(near, "P", txf_close * 0.90)
    if otm_put is not None and pd.notna(out["atm_put_premium_ratio_30d"]) and out["atm_put_premium_ratio_30d"] > 0:
        out["put_skew_proxy_30d"] = float((otm_put["mid"] / txf_close) / out["atm_put_premium_ratio_30d"])
    far_put = _nearest_contract(far, "P", txf_close)
    far_call = _nearest_contract(far, "C", txf_close)
    if far_put is not None and far_call is not None:
        far_straddle = float((far_put["mid"] + far_call["mid"]) / txf_close)
        if far_straddle > 0 and pd.notna(out["atm_straddle_premium_ratio_30d"]):
            out["iv_term_structure_proxy"] = float(out["atm_straddle_premium_ratio_30d"] / far_straddle)
    return out


def _nearest_contract(pool: pd.DataFrame, cp: str, target_strike: float) -> pd.Series | None:
    sub = pool[(pool["cp"].astype(str).str.upper() == cp) & pool["mid"].notna() & pool["strike"].notna()].copy()
    if sub.empty:
        return None
    sub["strike_distance"] = (sub["strike"] - target_strike).abs()
    return sub.sort_values(["strike_distance", "dte"]).iloc[0]


def _tradable_mask(options: pd.DataFrame) -> pd.Series:
    return (
        options["quote_quality_status"].fillna("").astype(str).eq("VALID")
        & _boolish_series(options["is_tradable_quote"])
        & options["bid"].notna()
        & options["ask"].notna()
        & (pd.to_numeric(options["ask"], errors="coerce") >= pd.to_numeric(options["bid"], errors="coerce"))
    )


def _boolish_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.fillna(False).astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y"])


def _contract_is_valid(contract: pd.Series | None) -> bool:
    if contract is None:
        return False
    return (
        str(contract.get("quote_quality_status", "")).upper() == "VALID"
        and str(contract.get("is_tradable_quote", False)).strip().lower() in {"true", "1", "yes", "y"}
        and pd.notna(contract.get("bid", np.nan))
        and pd.notna(contract.get("ask", np.nan))
        and float(contract.get("ask", np.nan)) >= float(contract.get("bid", np.nan))
    )


def _contract_value(contract: pd.Series | None, field: str, default: Any = np.nan) -> Any:
    if contract is None:
        return default
    return contract.get(field, default)


def _ensure_quality_columns(options: pd.DataFrame) -> None:
    if "quote_quality_status" not in options.columns:
        options["quote_quality_status"] = "UNKNOWN"
    if "is_tradable_quote" not in options.columns:
        options["is_tradable_quote"] = False


def _market_txf_close(market_by_date: pd.DataFrame, date: pd.Timestamp) -> float:
    if date not in market_by_date.index:
        return float("nan")
    row = market_by_date.loc[date]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    value = row.get("txf_close", row.get("tx_close", np.nan))
    return float(value) if pd.notna(value) else float("nan")


def _safe_ratio(numerator: float, denominator: float) -> float:
    if pd.isna(numerator) or pd.isna(denominator) or float(denominator) == 0.0:
        return float("nan")
    return float(numerator / denominator)


def _root_cause_markdown(audit: pd.DataFrame) -> str:
    daily = audit[audit["section"] == "daily_root_cause"] if not audit.empty and "section" in audit else pd.DataFrame()
    aggregate = audit[audit["section"] == "aggregate_summary"] if not audit.empty and "section" in audit else pd.DataFrame()
    lines = [
        "# Local TXO Proxy Missing Root-Cause Diagnostics",
        "",
        "This report diagnoses why local TXO-chain volatility proxies were missing around the 2020 crash window. It does not modify processed data, quote gates, formulas, weights, or trades.",
        "",
        "## Root Cause Distribution",
        "",
    ]
    cause_rows = aggregate[aggregate.get("metric", pd.Series(dtype=str)) == "root_cause_distribution"] if not aggregate.empty else pd.DataFrame()
    if cause_rows.empty:
        lines.append("- No root-cause rows.")
    else:
        for row in cause_rows.itertuples(index=False):
            lines.append(f"- {row.root_cause}: {row.days}")
    lines.extend(["", "## 2020 Options Coverage", ""])
    month_rows = aggregate[aggregate.get("metric", pd.Series(dtype=str)) == "options_rows_by_month_2020"] if not aggregate.empty else pd.DataFrame()
    if month_rows.empty:
        lines.append("- No 2020 monthly options rows.")
    else:
        for row in month_rows.itertuples(index=False):
            lines.append(f"- {row.month}: options_rows={row.options_rows}, tradable_options_rows={row.tradable_options_rows}")
    align = aggregate[aggregate.get("metric", pd.Series(dtype=str)) == "date_alignment_2020"] if not aggregate.empty else pd.DataFrame()
    if not align.empty:
        row = align.iloc[0]
        lines.extend(
            [
                "",
                "## Date Alignment",
                "",
                f"- market_dates={row.get('market_dates')}, options_dates={row.get('options_dates')}, aligned_dates={row.get('aligned_dates')}",
            ]
        )
    if not daily.empty:
        top = daily["root_cause"].fillna("UNKNOWN").astype(str).value_counts().idxmax()
        lines.extend(["", "## Diagnostic Summary", "", f"- Dominant root cause: {top}"])
    lines.extend(
        [
            "",
            "## Required Limitations",
            "",
            "- This audit does not fill missing data.",
            "- It does not relax quote quality requirements.",
            "- It does not change HedgeNeedScore weights or formulas.",
            "- It does not create trades.",
        ]
    )
    return "\n".join(lines) + "\n"


def _audit_markdown(audit: pd.DataFrame) -> str:
    lines = [
        "# Local Risk Indicator Build Audit",
        "",
        "This report audits local TXO-chain derived risk indicators. It does not create trades or change HedgeNeedScore weights.",
        "",
        "## Checks",
        "",
    ]
    if audit.empty:
        lines.append("- No audit rows.")
    else:
        for row in audit.head(80).itertuples(index=False):
            check = getattr(row, "check", "")
            status = getattr(row, "status", "")
            value = getattr(row, "value", "")
            detail = getattr(row, "detail", "")
            lines.append(f"- {check}: {status}, value={value}, detail={detail}")
    lines.extend(
        [
            "",
            "## Required Limitations",
            "",
            "- Local TXO-chain volatility fields are proxies, not official VIX.",
            "- Missing fields remain NaN.",
            "- No ranked parameter output is produced.",
        ]
    )
    return "\n".join(lines) + "\n"
