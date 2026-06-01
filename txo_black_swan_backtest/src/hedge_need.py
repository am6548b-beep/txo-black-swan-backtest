"""Risk-scaled hedge need diagnostics.

This module does not create trades. It maps fixed, coarse risk modules to a
target hedge coverage and compares that target with currently reported active
put-spread protection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data_loader import load_macro_factors, load_market, load_options, load_portfolio
from .indicators import add_market_indicators
from .macro_regime import add_macro_regime_indicators


RISK_WEIGHTS = {
    "ValuationRisk": 0.20,
    "VolatilityComplacencyRisk": 0.20,
    "MacroDemandFragility": 0.15,
    "SupplyStressIndex": 0.15,
    "LiquidityStress": 0.10,
    "TrendFragility": 0.20,
}

CRASH_WINDOWS: dict[str, tuple[str, str]] = {
    "2008": ("2008-01-01", "2008-12-31"),
    "2011": ("2011-01-01", "2011-12-31"),
    "2015": ("2015-06-01", "2015-12-31"),
    "2018": ("2018-01-01", "2018-12-31"),
    "2020": ("2020-01-01", "2020-12-31"),
    "2022": ("2022-01-01", "2022-12-31"),
}


def write_hedge_need_diagnostics(
    data_dir: Path,
    report_dir: Path,
    config: dict,
    put_params: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Write HedgeNeedScore, coverage timeline, and crash-window audit reports."""

    report_dir.mkdir(parents=True, exist_ok=True)
    market = load_market(data_dir)
    if market.empty:
        empty = pd.DataFrame([{"status": "FAIL", "detail": "market.csv unavailable"}])
        empty.to_csv(report_dir / "hedge_need_score.csv", index=False)
        empty.to_csv(report_dir / "hedge_coverage_timeline.csv", index=False)
        empty.to_csv(report_dir / "hedge_gap_audit.csv", index=False)
        (report_dir / "hedge_need_score.md").write_text(_markdown(empty, empty), encoding="utf-8")
        return empty, empty, empty

    market = add_market_indicators(market)
    market = add_macro_regime_indicators(market, load_macro_factors(data_dir))
    portfolio = load_portfolio(data_dir, market, config)
    load_config = config.copy()
    load_config["runtime_mode"] = "put_spread_only"
    options = load_options(data_dir, market, load_config)
    trades = _load_report_trades(report_dir)

    score = hedge_need_score_frame(market)
    coverage = hedge_coverage_timeline(score, portfolio, trades, config)
    coverage = _append_budget_and_attempt_flags(coverage, trades, config)
    audit = hedge_gap_audit(coverage, options, config, put_params)
    attribution = hedge_need_score_attribution(score, market)

    score.to_csv(report_dir / "hedge_need_score.csv", index=False)
    coverage.to_csv(report_dir / "hedge_coverage_timeline.csv", index=False)
    audit.to_csv(report_dir / "hedge_gap_audit.csv", index=False)
    attribution.to_csv(report_dir / "hedge_need_score_attribution.csv", index=False)
    (report_dir / "hedge_need_score.md").write_text(_markdown(score, audit), encoding="utf-8")
    (report_dir / "hedge_need_score_attribution.md").write_text(_attribution_markdown(attribution), encoding="utf-8")
    return score, coverage, audit


def hedge_need_score_frame(market: pd.DataFrame) -> pd.DataFrame:
    """Build fixed-weight hedge need scores from current and past data only."""

    out = market.sort_values("date").copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["ValuationRisk"] = pd.to_numeric(out.get("ValuationRiskIndex", 50.0), errors="coerce").fillna(50.0).clip(0, 100)
    out["VolatilityComplacencyRisk"] = (100.0 - pd.to_numeric(out.get("vix_percentile_3y", np.nan), errors="coerce")).clip(0, 100)
    out["VolatilityComplacencyRisk"] = out["VolatilityComplacencyRisk"].fillna(50.0)
    out["MacroDemandFragility"] = pd.to_numeric(out.get("MacroDemandFragilityIndex", 0.0), errors="coerce").fillna(0.0).clip(0, 100)
    out["SupplyStressIndex"] = pd.to_numeric(out.get("SupplyStressIndex", 0.0), errors="coerce").fillna(0.0).clip(0, 100)
    out["LiquidityStress"] = pd.to_numeric(out.get("LiquidityStressIndex", 50.0), errors="coerce").fillna(50.0).clip(0, 100)
    out["TrendFragility"] = _trend_fragility(out)
    out["HedgeNeedScore"] = sum(out[col] * weight for col, weight in RISK_WEIGHTS.items()).clip(0, 100)
    out["target_hedge_coverage"] = out["HedgeNeedScore"].map(target_hedge_coverage)
    out["vix_proxy_in_use"] = bool(out.get("vix_is_proxy", pd.Series([False])).any())
    return out[
        [
            "date",
            "HedgeNeedScore",
            "ValuationRisk",
            "VolatilityComplacencyRisk",
            "MacroDemandFragility",
            "SupplyStressIndex",
            "LiquidityStress",
            "TrendFragility",
            "target_hedge_coverage",
            "vix_proxy_in_use",
        ]
    ]


def target_hedge_coverage(score: float) -> float:
    """Map HedgeNeedScore to target coverage using fixed coarse buckets."""

    if pd.isna(score):
        return 0.0
    value = float(np.clip(score, 0.0, 100.0))
    if value <= 25.0:
        return _linear(value, 0.0, 25.0, 0.0, 0.10)
    if value <= 50.0:
        return _linear(value, 25.0, 50.0, 0.10, 0.25)
    if value <= 75.0:
        return _linear(value, 50.0, 75.0, 0.25, 0.50)
    return _linear(value, 75.0, 100.0, 0.50, 0.80)


def hedge_coverage_timeline(score: pd.DataFrame, portfolio: pd.DataFrame, trades: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Compare target hedge coverage with active reported put-spread protection."""

    out = score.sort_values("date").copy()
    port = portfolio.copy()
    port["date"] = pd.to_datetime(port["date"], errors="coerce")
    out = out.merge(port[["date", "stock_equity", "portfolio_beta"]], on="date", how="left")
    out["stock_equity"] = out["stock_equity"].ffill().fillna(float(config.get("initial_stock_equity", 0.0)))
    out["portfolio_beta"] = out["portfolio_beta"].ffill().fillna(float(config.get("portfolio_beta", 1.0)))
    out["portfolio_beta_exposure"] = out["stock_equity"] * out["portfolio_beta"]
    positions = _put_spread_positions(trades, config)
    protections = []
    active_counts = []
    active_ids = []
    for date in out["date"]:
        active = [pos for pos in positions if pos["entry_date"] <= date <= pos["exit_date"]]
        protections.append(float(sum(pos["max_protection"] for pos in active)))
        active_counts.append(int(len(active)))
        active_ids.append(";".join(pos["position_id"] for pos in active))
    out["active_put_spread_max_protection"] = protections
    out["active_put_spread_count"] = active_counts
    out["active_position_id"] = active_ids
    out["current_hedge_coverage"] = np.where(
        out["portfolio_beta_exposure"] > 0,
        out["active_put_spread_max_protection"] / out["portfolio_beta_exposure"],
        0.0,
    )
    out["hedge_gap"] = out["target_hedge_coverage"] - out["current_hedge_coverage"]
    return out


def hedge_gap_audit(coverage: pd.DataFrame, options: pd.DataFrame, config: dict, put_params: dict) -> pd.DataFrame:
    """Audit whether hedge need rose before historical crash windows."""

    rows: list[dict[str, Any]] = []
    for label, (start_text, end_text) in CRASH_WINDOWS.items():
        start = pd.Timestamp(start_text)
        pre_start = start - pd.Timedelta(days=180)
        pre = coverage[(coverage["date"] >= pre_start) & (coverage["date"] < start)].copy()
        if pre.empty:
            rows.append({"section": "crash_window_audit", "crash_window": label, "status": "WARN", "reason_gap_not_closed": "NO_MARKET_ROWS"})
            continue
        score_change = float(pre["HedgeNeedScore"].iloc[-1] - pre["HedgeNeedScore"].iloc[0])
        target_change = float(pre["target_hedge_coverage"].iloc[-1] - pre["target_hedge_coverage"].iloc[0])
        max_target = float(pre["target_hedge_coverage"].max())
        max_current = float(pre["current_hedge_coverage"].max())
        should_days = pre[pre["should_attempt_hedge"].astype(bool)]
        blocked_budget = int((pre["if_not_attempted_reason"] == "BUDGET_BLOCKED").sum())
        availability = _availability_summary(should_days, options, put_params, config)
        reason = _gap_reason(score_change, max_target, max_current, blocked_budget, availability)
        rows.append(
            {
                "section": "crash_window_audit",
                "crash_window": label,
                "crash_start": start_text,
                "crash_end": end_text,
                "pre_window_days": int(len(pre)),
                "hedge_need_score_start": float(pre["HedgeNeedScore"].iloc[0]),
                "hedge_need_score_end": float(pre["HedgeNeedScore"].iloc[-1]),
                "hedge_need_score_change": score_change,
                "target_hedge_coverage_start": float(pre["target_hedge_coverage"].iloc[0]),
                "target_hedge_coverage_end": float(pre["target_hedge_coverage"].iloc[-1]),
                "target_hedge_coverage_change": target_change,
                "max_target_hedge_coverage": max_target,
                "max_current_hedge_coverage": max_current,
                "actual_current_hedge_coverage_followed_target": bool(max_current + 1e-9 >= max_target * 0.75),
                "should_attempt_days": int(len(should_days)),
                "budget_blocked_days": blocked_budget,
                **availability,
                "reason_gap_not_closed": reason,
            }
        )
    rows.append(
        {
            "section": "overfitting_controls",
            "status": "PASS",
            "detail": "fixed module weights; fixed target mapping; no crash-window tuning; no parameter search",
        }
    )
    return pd.DataFrame(rows)


def hedge_need_score_attribution(score: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    """Explain fixed-weight module contributions and pre-crash score behavior."""

    daily = _daily_attribution(score, market)
    rows: list[dict[str, Any]] = daily.to_dict("records")
    rows.extend(_crash_attribution_rows(daily))
    rows.extend(_false_calm_rows(daily, market))
    rows.append(
        {
            "section": "overfitting_controls",
            "status": "PASS",
            "detail": "attribution uses existing fixed weights and existing score columns only",
        }
    )
    return pd.DataFrame(rows)


def _daily_attribution(score: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    out = score.copy()
    macro_missing = _macro_missing_count(market)
    out = out.merge(macro_missing, on="date", how="left")
    out["macro_proxy_missing_count"] = out["macro_proxy_missing_count"].fillna(0).astype(int)
    for module, weight in RISK_WEIGHTS.items():
        out[f"{module}_weighted_contribution"] = out[module].astype(float) * weight
    out["weighted_sum_check"] = sum(out[f"{module}_weighted_contribution"] for module in RISK_WEIGHTS)
    out["score_confidence"] = out.apply(_score_confidence, axis=1)
    out["section"] = "daily_attribution"
    cols = [
        "section",
        "date",
        "HedgeNeedScore",
        *RISK_WEIGHTS.keys(),
        *(f"{module}_weighted_contribution" for module in RISK_WEIGHTS),
        "weighted_sum_check",
        "vix_proxy_in_use",
        "macro_proxy_missing_count",
        "score_confidence",
    ]
    return out[list(cols)]


def _crash_attribution_rows(daily: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    work = daily.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    modules = list(RISK_WEIGHTS.keys())
    for label, (start_text, _) in CRASH_WINDOWS.items():
        start = pd.Timestamp(start_text)
        pre = work[(work["date"] >= start - pd.Timedelta(days=180)) & (work["date"] < start)].copy()
        if pre.empty:
            rows.append({"section": "crash_pre_window_attribution", "crash_window": label, "status": "WARN", "detail": "no pre-window rows"})
            continue
        first = pre.iloc[0]
        last = pre.iloc[-1]
        changes = {module: float(last[module] - first[module]) for module in modules}
        weighted_changes = {module: float(changes[module] * RISK_WEIGHTS[module]) for module in modules}
        positive = [module for module, value in sorted(weighted_changes.items(), key=lambda item: item[1], reverse=True) if value > 0.5]
        negative = [module for module, value in sorted(weighted_changes.items(), key=lambda item: item[1]) if value < -0.5]
        flat = [module for module, value in changes.items() if abs(value) <= 1.0]
        proxy_modules = ["VolatilityComplacencyRisk"] if bool(pre["vix_proxy_in_use"].astype(bool).any()) else []
        if pd.to_numeric(pre["macro_proxy_missing_count"], errors="coerce").max() > 0:
            proxy_modules.extend(["MacroDemandFragility", "SupplyStressIndex", "LiquidityStress"])
        row = {
            "section": "crash_pre_window_attribution",
            "crash_window": label,
            "score_start": float(first["HedgeNeedScore"]),
            "score_end": float(last["HedgeNeedScore"]),
            "score_change": float(last["HedgeNeedScore"] - first["HedgeNeedScore"]),
            "max_score": float(pre["HedgeNeedScore"].max()),
            "min_score": float(pre["HedgeNeedScore"].min()),
            "top_positive_contributors": ";".join(positive),
            "top_negative_contributors": ";".join(negative),
            "modules_that_stayed_flat": ";".join(flat),
            "modules_with_proxy_or_missing_data": ";".join(dict.fromkeys(proxy_modules)),
            "score_confidence_min": _min_confidence(pre["score_confidence"]),
        }
        for module in modules:
            row[f"{module}_start"] = float(first[module])
            row[f"{module}_end"] = float(last[module])
            row[f"{module}_change"] = float(changes[module])
            row[f"{module}_weighted_change"] = float(weighted_changes[module])
        rows.append(row)
    return rows


def _false_calm_rows(daily: pd.DataFrame, market: pd.DataFrame) -> list[dict[str, Any]]:
    work = daily.merge(
        market[["date", "tx_close", "drawdown_20d_from_high"]].copy(),
        on="date",
        how="left",
    )
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work["ret"] = pd.to_numeric(work["tx_close"], errors="coerce").pct_change()
    work["realized_vol_20d"] = work["ret"].rolling(20, min_periods=20).std() * np.sqrt(252) * 100
    work["score_20d_change"] = work["HedgeNeedScore"].diff(20)
    work["rv_20d_change"] = work["realized_vol_20d"].diff(20)
    work["drawdown_20d_change"] = (-pd.to_numeric(work["drawdown_20d_from_high"], errors="coerce")).diff(20)
    work["trend_20d_change"] = work["TrendFragility"].diff(20)
    work["liquidity_20d_change"] = work["LiquidityStress"].diff(20)
    checks = {
        "realized_volatility_rising_score_not_rising": (work["rv_20d_change"] > 5.0) & (work["score_20d_change"] <= 0.0),
        "drawdown_increasing_score_not_rising": (work["drawdown_20d_change"] > 0.05) & (work["score_20d_change"] <= 0.0),
        "trend_fragility_rising_score_muted": (work["trend_20d_change"] > 10.0) & (work["score_20d_change"] < 2.0),
        "liquidity_stress_rising_score_muted": (work["liquidity_20d_change"] > 10.0) & (work["score_20d_change"] < 2.0),
    }
    rows = []
    for name, mask in checks.items():
        flagged = work[mask.fillna(False)]
        rows.append(
            {
                "section": "false_calm_diagnostic",
                "check": name,
                "flagged_days": int(len(flagged)),
                "first_flag_date": str(flagged["date"].min().date()) if not flagged.empty else "",
                "last_flag_date": str(flagged["date"].max().date()) if not flagged.empty else "",
                "status": "WARN" if not flagged.empty else "PASS",
            }
        )
    return rows


def _append_budget_and_attempt_flags(coverage: pd.DataFrame, trades: pd.DataFrame, config: dict) -> pd.DataFrame:
    out = coverage.copy()
    spend = _annual_entry_spend_by_date(trades)
    max_budget_pct = float(config.get("max_annual_hedge_budget_pct", 0.03))
    threshold = float(config.get("hedge_gap_attempt_threshold", 0.05))
    used_values = []
    remaining_values = []
    for row in out.itertuples(index=False):
        date = pd.Timestamp(row.date)
        used = spend[(spend["date"] <= date) & (spend["year"] == date.year)]["entry_spend"].sum() if not spend.empty else 0.0
        limit = max_budget_pct * float(row.stock_equity)
        used_values.append(float(used))
        remaining_values.append(float(max(0.0, limit - used)))
    out["annual_budget_used"] = used_values
    out["annual_budget_limit"] = out["stock_equity"] * max_budget_pct
    out["annual_budget_remaining"] = remaining_values
    out["should_attempt_hedge"] = (out["hedge_gap"] > threshold) & (out["annual_budget_remaining"] > 0)
    out["if_not_attempted_reason"] = np.where(
        out["should_attempt_hedge"],
        "",
        np.where(out["hedge_gap"] <= threshold, "HEDGE_GAP_BELOW_THRESHOLD", "BUDGET_BLOCKED"),
    )
    return out


def _macro_missing_count(market: pd.DataFrame) -> pd.DataFrame:
    proxy_cols = [
        "MacroDemandFragilityIndex",
        "SupplyStressIndex",
        "LiquidityStressIndex",
        "ValuationRiskIndex",
    ]
    out = market[["date"]].copy()
    existing = [col for col in proxy_cols if col in market.columns]
    if existing:
        out["macro_proxy_missing_count"] = market[existing].isna().sum(axis=1)
    else:
        out["macro_proxy_missing_count"] = len(proxy_cols)
    return out


def _score_confidence(row: pd.Series) -> str:
    missing = int(row.get("macro_proxy_missing_count", 0))
    vix_proxy = bool(row.get("vix_proxy_in_use", False))
    if vix_proxy or missing >= 3:
        return "LOW"
    if missing > 0:
        return "MEDIUM"
    return "HIGH"


def _min_confidence(values: pd.Series) -> str:
    ranks = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    clean = values.dropna().astype(str)
    if clean.empty:
        return "LOW"
    return min(clean, key=lambda item: ranks.get(item, 0))


def _attribution_markdown(attribution: pd.DataFrame) -> str:
    crash = attribution[attribution["section"] == "crash_pre_window_attribution"] if not attribution.empty else pd.DataFrame()
    confidence = attribution[attribution["section"] == "daily_attribution"]["score_confidence"].value_counts() if not attribution.empty and "score_confidence" in attribution else pd.Series(dtype=int)
    false_calm = attribution[attribution["section"] == "false_calm_diagnostic"] if not attribution.empty else pd.DataFrame()
    lines = [
        "# Hedge Need Score Attribution",
        "",
        "This report explains fixed-weight HedgeNeedScore contributions. It does not change weights, score mapping, or trading rules.",
        "",
        "## Score Confidence",
        "",
    ]
    if confidence.empty:
        lines.append("- No confidence rows.")
    else:
        for label, count in confidence.items():
            lines.append(f"- {label}: {int(count)}")
    lines.extend(["", "## Crash Pre-Window Attribution", ""])
    if crash.empty:
        lines.append("- No crash attribution rows.")
    else:
        for row in crash.itertuples(index=False):
            lines.append(
                f"- {row.crash_window}: score_change={row.score_change}, "
                f"positive={row.top_positive_contributors}, negative={row.top_negative_contributors}, "
                f"flat={row.modules_that_stayed_flat}, proxy={row.modules_with_proxy_or_missing_data}"
            )
    lines.extend(["", "## False Calm Diagnostics", ""])
    if false_calm.empty:
        lines.append("- No false calm rows.")
    else:
        for row in false_calm.itertuples(index=False):
            lines.append(f"- {row.check}: {row.status}, flagged_days={row.flagged_days}")
    lines.extend(
        [
            "",
            "## Required Limitations",
            "",
            "- This report is attribution only.",
            "- It does not alter fixed weights or target coverage mapping.",
            "- It does not infer rules from crash windows.",
            "- It is not an investment conclusion.",
        ]
    )
    return "\n".join(lines) + "\n"


def _trend_fragility(market: pd.DataFrame) -> pd.Series:
    close = pd.to_numeric(market["tx_close"], errors="coerce")
    ma200 = pd.to_numeric(market.get("ma200", np.nan), errors="coerce")
    ret_126d = pd.to_numeric(market.get("ret_126d", np.nan), errors="coerce")
    dd20 = pd.to_numeric(market.get("drawdown_20d_from_high", 0.0), errors="coerce")
    ma_score = _scale_series(-(close / ma200.replace(0, np.nan) - 1.0), -0.05, 0.15)
    ret_score = _scale_series(-ret_126d, -0.05, 0.20)
    dd_score = _scale_series(-dd20, 0.03, 0.18)
    return (0.4 * ma_score + 0.3 * ret_score + 0.3 * dd_score).fillna(50.0).clip(0, 100)


def _put_spread_positions(trades: pd.DataFrame, config: dict) -> list[dict[str, Any]]:
    if trades.empty:
        return []
    t = trades.copy()
    t["date"] = pd.to_datetime(t["date"], errors="coerce")
    t["expiry"] = pd.to_datetime(t["expiry"], errors="coerce")
    positions: list[dict[str, Any]] = []
    point_value = float(config.get("txo_point_value", 50.0))
    for pid, group in t[t.get("strategy", "") == "put_spread"].groupby("position_id"):
        opens = group[group["reason"].astype(str).str.contains("open|quarterly|rolling", case=False, regex=True, na=False)]
        if opens.empty:
            continue
        long_leg = opens[opens["action"].astype(str).str.upper() == "BUY"]
        short_leg = opens[opens["action"].astype(str).str.upper() == "SELL"]
        if long_leg.empty or short_leg.empty:
            continue
        exit_rows = group[~group.index.isin(opens.index)]
        entry_date = opens["date"].min()
        expiry = opens["expiry"].min()
        exit_date = exit_rows["date"].max() if not exit_rows.empty else expiry
        qty = int(abs(pd.to_numeric(long_leg.iloc[0].get("quantity", 0), errors="coerce")))
        long_strike = float(long_leg.iloc[0]["strike"])
        short_strike = float(short_leg.iloc[0]["strike"])
        positions.append(
            {
                "position_id": str(pid),
                "entry_date": entry_date,
                "exit_date": exit_date,
                "expiry": expiry,
                "max_protection": max(0.0, long_strike - short_strike) * point_value * qty,
            }
        )
    return positions


def _annual_entry_spend_by_date(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["date", "year", "entry_spend"])
    t = trades.copy()
    t["date"] = pd.to_datetime(t["date"], errors="coerce")
    opens = t[(t.get("strategy", "") == "put_spread") & t["reason"].astype(str).str.contains("open|quarterly|rolling", case=False, regex=True, na=False)].copy()
    if opens.empty:
        return pd.DataFrame(columns=["date", "year", "entry_spend"])
    opens["entry_spend"] = -pd.to_numeric(opens["cash_flow"], errors="coerce").fillna(0.0)
    opens["year"] = opens["date"].dt.year
    return opens[["date", "year", "entry_spend"]]


def _availability_summary(days: pd.DataFrame, options: pd.DataFrame, put_params: dict, config: dict) -> dict[str, Any]:
    if days.empty or options.empty:
        return {
            "contract_available_days": 0,
            "quote_or_liquidity_blocked_days": 0,
            "contract_unavailable_days": int(len(days)),
        }
    opt = options.copy()
    opt["date"] = pd.to_datetime(opt["date"], errors="coerce")
    by_date = {pd.Timestamp(date): group.copy() for date, group in opt.groupby("date", sort=False)}
    dte_min = int(put_params.get("target_dte_min", 60))
    dte_max = int(put_params.get("target_dte_max", 120))
    valid = blocked = unavailable = 0
    min_volume = float(config.get("wide_spread_volume_threshold", 50))
    min_oi = float(config.get("min_open_interest", 100))
    for row in days.itertuples(index=False):
        chain = by_date.get(pd.Timestamp(row.date), pd.DataFrame())
        puts = chain[(chain.get("cp", "") == "P") & (chain.get("dte", pd.Series(dtype=float)).between(dte_min, dte_max))] if not chain.empty else pd.DataFrame()
        if puts.empty:
            unavailable += 1
            continue
        tradable = puts[
            puts.get("is_tradable_quote", pd.Series(False, index=puts.index)).astype(bool)
            & (puts.get("quote_quality_status", pd.Series("", index=puts.index)).astype(str) == "VALID")
            & (pd.to_numeric(puts.get("volume", 0), errors="coerce") >= min_volume)
            & (pd.to_numeric(puts.get("open_interest", 0), errors="coerce") >= min_oi)
        ]
        if tradable.empty:
            blocked += 1
        else:
            valid += 1
    return {
        "contract_available_days": int(valid),
        "quote_or_liquidity_blocked_days": int(blocked),
        "contract_unavailable_days": int(unavailable),
    }


def _gap_reason(score_change: float, max_target: float, max_current: float, budget_blocked: int, availability: dict[str, Any]) -> str:
    if max_target <= 0.10 or score_change <= 0:
        return "SCORE_DID_NOT_RISE"
    if budget_blocked > 0:
        return "BUDGET_BLOCKED"
    if availability.get("contract_unavailable_days", 0) > availability.get("contract_available_days", 0):
        return "CONTRACT_AVAILABILITY_LIMIT"
    if availability.get("quote_or_liquidity_blocked_days", 0) > availability.get("contract_available_days", 0):
        return "QUOTE_OR_LIQUIDITY_LIMIT"
    if max_current + 1e-9 < max_target * 0.75:
        return "COVERAGE_GAP"
    return "COVERAGE_TRACKED_TARGET"


def _load_report_trades(report_dir: Path) -> pd.DataFrame:
    path = report_dir / "trades.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _markdown(score: pd.DataFrame, audit: pd.DataFrame) -> str:
    lines = [
        "# Hedge Need Diagnostics",
        "",
        "This report maps fixed risk modules to target hedge coverage. It does not create trades, rank variants, optimize parameters, or change execution rules.",
        "",
        "## Fixed Weights",
        "",
    ]
    for name, weight in RISK_WEIGHTS.items():
        lines.append(f"- {name}: {weight:.2f}")
    if not score.empty and "vix_proxy_in_use" in score:
        lines.extend(["", "## Data Flags", "", f"- vix_proxy_in_use: {bool(score['vix_proxy_in_use'].any())}"])
    crash = audit[audit.get("section", pd.Series(dtype=str)) == "crash_window_audit"] if not audit.empty else pd.DataFrame()
    lines.extend(["", "## Crash Window Audit", ""])
    if crash.empty:
        lines.append("- No crash audit rows.")
    else:
        for row in crash.itertuples(index=False):
            lines.append(
                f"- {row.crash_window}: score_change={row.hedge_need_score_change}, "
                f"target_change={row.target_hedge_coverage_change}, reason={row.reason_gap_not_closed}"
            )
    lines.extend(
        [
            "",
            "## Required Limitations",
            "",
            "- No crash-window tuning is used.",
            "- No automatic weight search is used.",
            "- Score mapping is fixed before diagnostics.",
            "- This is not an investment conclusion.",
        ]
    )
    return "\n".join(lines) + "\n"


def _linear(value: float, x0: float, x1: float, y0: float, y1: float) -> float:
    return float(y0 + (value - x0) / (x1 - x0) * (y1 - y0))


def _scale_series(series: pd.Series, low: float, high: float) -> pd.Series:
    return ((series - low) / (high - low) * 100.0).clip(0, 100)
