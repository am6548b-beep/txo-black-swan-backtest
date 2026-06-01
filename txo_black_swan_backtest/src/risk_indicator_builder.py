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
    indicators.to_csv(processed_dir / "risk_indicators.csv", index=False)
    audit.to_csv(report_dir / "local_risk_indicator_build_audit.csv", index=False)
    (report_dir / "local_risk_indicator_build_audit.md").write_text(_audit_markdown(audit), encoding="utf-8")
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
        & options["is_tradable_quote"].astype(bool)
        & options["bid"].notna()
        & options["ask"].notna()
        & (pd.to_numeric(options["ask"], errors="coerce") >= pd.to_numeric(options["bid"], errors="coerce"))
    )


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
