"""Coverage diagnostics for put-spread crash-window gaps.

The functions here read generated reports and processed market/options data.
They do not change strategy rules, parameters, fills, or backtest results.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data_loader import load_macro_factors, load_market, load_options
from .indicators import add_market_indicators
from .macro_regime import add_macro_regime_indicators, macro_restrictions
from .utils import year_key


CRASH_WINDOWS: dict[str, tuple[str, str]] = {
    "2008": ("2008-01-01", "2008-12-31"),
    "2011": ("2011-01-01", "2011-12-31"),
    "2015": ("2015-06-01", "2015-12-31"),
    "2018": ("2018-01-01", "2018-12-31"),
    "2020": ("2020-01-01", "2020-12-31"),
    "2022": ("2022-01-01", "2022-12-31"),
}

REJECTION_ORDER = [
    "BELOW_MA200",
    "RETURN_NOT_HIGH_ENOUGH",
    "VIX_NOT_LOW_ENOUGH",
    "VIX_PROXY_IN_USE",
    "EVENT_FLAG_BLOCKED",
    "BUDGET_EXCEEDED",
    "EXISTING_POSITION",
    "NO_EXPIRY_IN_DTE_RANGE",
    "NO_LONG_PUT_CANDIDATE",
    "NO_SHORT_PUT_CANDIDATE",
    "QUOTE_NOT_VALID",
    "LOW_VOLUME",
    "LOW_OPEN_INTEREST",
    "BAD_EXPIRY",
    "UNKNOWN",
]


@dataclass(frozen=True)
class PositionWindow:
    position_id: str
    entry_date: pd.Timestamp
    expiry: pd.Timestamp
    exit_date: pd.Timestamp


def write_put_spread_coverage_audit(
    report_dir: Path,
    data_dir: Path,
    config: dict,
    put_params: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write coverage audit CSV/MD and entry-signal daily audit CSV."""

    report_dir.mkdir(parents=True, exist_ok=True)
    trades = _read_csv(report_dir / "trades.csv")
    equity = _read_csv(report_dir / "equity_curve.csv")
    lifecycle = _read_csv(report_dir / "position_lifecycle_audit.csv")
    market = load_market(data_dir)
    if market.empty:
        coverage = pd.DataFrame([_row("data", "market_available", "FAIL", "market.csv unavailable")])
        entry = pd.DataFrame()
        coverage.to_csv(report_dir / "put_spread_coverage_audit.csv", index=False)
        entry.to_csv(report_dir / "put_spread_entry_signal_audit.csv", index=False)
        (report_dir / "put_spread_coverage_audit.md").write_text(_markdown(coverage, entry), encoding="utf-8")
        return coverage, entry

    market = add_market_indicators(market)
    market = add_macro_regime_indicators(market, load_macro_factors(data_dir))
    load_config = config.copy()
    load_config["runtime_mode"] = "put_spread_only"
    options = load_options(data_dir, market, load_config)
    options_by_date = {pd.Timestamp(date): group.copy() for date, group in options.groupby("date", sort=False)}

    positions = _position_windows(trades, lifecycle)
    entry = _entry_signal_audit(market, options_by_date, positions, trades, config, put_params)
    coverage = pd.concat(
        [
            _crash_coverage_rows(market, equity, positions),
            _rejection_summary_rows(entry),
            _pre_window_rows(entry),
            _forced_unfilled_rows(trades, lifecycle, options, config, put_params),
        ],
        ignore_index=True,
    )
    coverage.to_csv(report_dir / "put_spread_coverage_audit.csv", index=False)
    entry.to_csv(report_dir / "put_spread_entry_signal_audit.csv", index=False)
    (report_dir / "put_spread_coverage_audit.md").write_text(_markdown(coverage, entry), encoding="utf-8")
    return coverage, entry


def _entry_signal_audit(
    market: pd.DataFrame,
    options_by_date: dict[pd.Timestamp, pd.DataFrame],
    positions: list[PositionWindow],
    trades: pd.DataFrame,
    config: dict,
    put_params: dict,
) -> pd.DataFrame:
    annual_spend = _annual_spend_by_date(trades)
    rows: list[dict[str, Any]] = []
    for row in market.itertuples(index=False):
        date = pd.Timestamp(row.date)
        macro_state = str(getattr(row, "macro_state", "NORMAL"))
        restriction = macro_restrictions(macro_state)
        vix_threshold = 40.0 if macro_state == "STAGFLATION_PRESSURE" and float(getattr(row, "ValuationRiskIndex", 0.0)) > 70.0 else 30.0
        ma_ok = _bool_not_nan(float(row.tx_close) > float(row.ma200) if pd.notna(row.ma200) else np.nan)
        ret_ok = _bool_not_nan(float(row.ret_126d) > 0.20 if pd.notna(row.ret_126d) else np.nan)
        vix_ok = _bool_not_nan(float(row.vix_percentile_3y) < vix_threshold if pd.notna(row.vix_percentile_3y) else np.nan)
        event_ok = int(getattr(row, "event_flag", 0)) == 0
        active = _active_positions(positions, date)
        no_existing = not active
        budget_cap = float(config["max_annual_hedge_budget_pct"]) * float(restriction["hedge_budget_multiplier"]) * float(config["initial_stock_equity"])
        used = _spend_used_before(annual_spend, date)
        budget_ok = used < budget_cap
        candidates = _candidate_audit(row, options_by_date.get(date, pd.DataFrame()), config, put_params)
        market_conditions = bool(ma_ok and ret_ok and vix_ok and event_ok)
        reasons = _rejection_reasons(
            row,
            ma_ok,
            ret_ok,
            vix_ok,
            event_ok,
            budget_ok,
            no_existing,
            candidates,
        )
        rows.append(
            {
                "date": str(date.date()),
                "tx_close": float(row.tx_close),
                "txf_close": float(row.txf_close),
                "ma200": _float_or_blank(row.ma200),
                "ret_126d": _float_or_blank(row.ret_126d),
                "vix_percentile_3y": _float_or_blank(row.vix_percentile_3y),
                "vix_is_proxy": bool(getattr(row, "vix_is_proxy", False)),
                "tx_close_gt_ma200": ma_ok,
                "return_126d_gt_threshold": ret_ok,
                "vix_percentile_lt_threshold": vix_ok,
                "vix_threshold_used": vix_threshold,
                "event_flag_eq_0": event_ok,
                "annual_hedge_budget_available": budget_ok,
                "annual_hedge_budget_used": used,
                "annual_hedge_budget_cap": budget_cap,
                "no_existing_put_spread": no_existing,
                "active_position_ids": ";".join(active),
                "market_conditions_met": market_conditions,
                **candidates,
                "would_pass_entry_checks": bool(market_conditions and budget_ok and no_existing and candidates["both_legs_tradable"]),
                "rejection_reasons": ";".join(reasons),
                "primary_rejection_reason": reasons[0] if reasons else "",
            }
        )
    return pd.DataFrame(rows)


def _candidate_audit(row: Any, chain: pd.DataFrame, config: dict, put_params: dict) -> dict[str, Any]:
    date = pd.Timestamp(row.date)
    dte_min = int(put_params["target_dte_min"])
    macro_state = str(getattr(row, "macro_state", "NORMAL"))
    dte_max = int(put_params["target_dte_max"]) + int(macro_restrictions(macro_state)["extend_put_dte"])
    low_vix = pd.notna(row.vix_percentile_3y) and float(row.vix_percentile_3y) < 20.0
    long_m = put_params["long_put_moneyness_low_vix"] if low_vix else put_params["long_put_moneyness"]
    short_m = put_params["short_put_moneyness_low_vix"] if low_vix else put_params["short_put_moneyness"]
    target_long = float(row.txf_close) * float(long_m)
    target_short = float(row.txf_close) * float(short_m)
    puts_in_dte = chain[(chain["cp"] == "P") & (chain["dte"].between(dte_min, dte_max))].copy() if not chain.empty else pd.DataFrame()
    dte_range = not puts_in_dte.empty
    long_raw = _nearest_raw(puts_in_dte, target_long, dte_min, dte_max)
    long_selected = _select_liquid_nearest(puts_in_dte, target_long, dte_min, dte_max, config)
    if long_selected is not None:
        short_pool = puts_in_dte[puts_in_dte["expiry"] == pd.Timestamp(long_selected["expiry"])].copy()
    elif long_raw is not None and "expiry" in long_raw:
        short_pool = puts_in_dte[puts_in_dte["expiry"] == pd.Timestamp(long_raw["expiry"])].copy()
    else:
        short_pool = puts_in_dte.iloc[0:0].copy()
    short_raw = _nearest_raw(short_pool, target_short, dte_min, dte_max)
    short_selected = _select_liquid_nearest(short_pool, target_short, dte_min, dte_max, config) if long_selected is not None else None
    if short_selected is not None and long_selected is not None and float(short_selected["strike"]) >= float(long_selected["strike"]):
        short_selected = None
    raw_rows = [raw for raw in [long_raw, short_raw] if raw is not None]
    quote_valid = bool(raw_rows) and all(str(raw.get("quote_quality_status", "VALID")) == "VALID" and _as_bool(raw.get("is_tradable_quote", True)) for raw in raw_rows)
    low_volume = any(float(raw.get("volume", 0.0)) < float(config.get("wide_spread_volume_threshold", 50)) for raw in raw_rows)
    low_oi = any(float(raw.get("open_interest", 0.0)) < float(config.get("min_open_interest", 100)) for raw in raw_rows)
    bad_expiry = bool(raw_rows) and any(pd.isna(raw.get("expiry")) or float(raw.get("dte", -1)) < 0 for raw in raw_rows)
    return {
        "target_dte_min": dte_min,
        "target_dte_max": dte_max,
        "target_long_put_moneyness": float(long_m),
        "target_short_put_moneyness": float(short_m),
        "target_long_put_strike": target_long,
        "target_short_put_strike": target_short,
        "dte_range_matched": dte_range,
        "candidate_long_put_found": long_raw is not None,
        "candidate_short_put_found": short_raw is not None,
        "long_put_selected": long_selected is not None,
        "short_put_selected": short_selected is not None,
        "both_legs_tradable": bool(long_selected is not None and short_selected is not None),
        "quote_quality_valid": quote_valid,
        "liquidity_filter_passed": bool(long_selected is not None and short_selected is not None),
        "low_volume_block": low_volume,
        "low_open_interest_block": low_oi,
        "bad_expiry_block": bad_expiry,
        "long_raw_quote_quality_status": "" if long_raw is None else str(long_raw.get("quote_quality_status", "")),
        "short_raw_quote_quality_status": "" if short_raw is None else str(short_raw.get("quote_quality_status", "")),
        "long_raw_volume": "" if long_raw is None else _float_or_blank(long_raw.get("volume")),
        "short_raw_volume": "" if short_raw is None else _float_or_blank(short_raw.get("volume")),
        "long_raw_open_interest": "" if long_raw is None else _float_or_blank(long_raw.get("open_interest")),
        "short_raw_open_interest": "" if short_raw is None else _float_or_blank(short_raw.get("open_interest")),
    }


def _nearest_raw(pool: pd.DataFrame, target_strike: float, dte_min: int, dte_max: int) -> dict[str, Any] | None:
    if pool.empty:
        return None
    tmp = pool.copy()
    tmp["expiry_score"] = (tmp["dte"] - (dte_min + dte_max) / 2.0).abs()
    tmp["strike_score"] = (tmp["strike"] - target_strike).abs()
    return tmp.sort_values(["expiry_score", "strike_score"]).iloc[0].to_dict()


def _select_liquid_nearest(pool: pd.DataFrame, target_strike: float, dte_min: int, dte_max: int, config: dict) -> dict[str, Any] | None:
    if pool.empty:
        return None
    min_volume = float(config.get("wide_spread_volume_threshold", 50))
    min_oi = float(config.get("min_open_interest", 100))
    liquid = pool[
        pool["tradable"].astype(bool)
        & pool["is_tradable_quote"].astype(bool)
        & (pool["volume"] >= min_volume)
        & (pool["open_interest"] >= min_oi)
        & (pool["ask"] >= pool["bid"])
        & (pool["bid"] >= 0)
    ].copy()
    if liquid.empty:
        return None
    liquid["expiry_score"] = (liquid["dte"] - (dte_min + dte_max) / 2.0).abs()
    liquid["strike_score"] = (liquid["strike"] - target_strike).abs()
    return liquid.sort_values(["expiry_score", "strike_score"]).iloc[0].to_dict()


def _rejection_reasons(
    row: Any,
    ma_ok: bool,
    ret_ok: bool,
    vix_ok: bool,
    event_ok: bool,
    budget_ok: bool,
    no_existing: bool,
    candidates: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    if not ma_ok:
        reasons.append("BELOW_MA200")
    if not ret_ok:
        reasons.append("RETURN_NOT_HIGH_ENOUGH")
    if not vix_ok:
        reasons.append("VIX_NOT_LOW_ENOUGH")
    if bool(getattr(row, "vix_is_proxy", False)):
        reasons.append("VIX_PROXY_IN_USE")
    if not event_ok:
        reasons.append("EVENT_FLAG_BLOCKED")
    if not budget_ok:
        reasons.append("BUDGET_EXCEEDED")
    if not no_existing:
        reasons.append("EXISTING_POSITION")
    if not candidates["dte_range_matched"]:
        reasons.append("NO_EXPIRY_IN_DTE_RANGE")
    if not candidates["candidate_long_put_found"]:
        reasons.append("NO_LONG_PUT_CANDIDATE")
    if not candidates["candidate_short_put_found"]:
        reasons.append("NO_SHORT_PUT_CANDIDATE")
    if candidates["candidate_long_put_found"] or candidates["candidate_short_put_found"]:
        if not candidates["quote_quality_valid"]:
            reasons.append("QUOTE_NOT_VALID")
        if candidates["low_volume_block"]:
            reasons.append("LOW_VOLUME")
        if candidates["low_open_interest_block"]:
            reasons.append("LOW_OPEN_INTEREST")
        if candidates["bad_expiry_block"]:
            reasons.append("BAD_EXPIRY")
    if not reasons and not candidates["both_legs_tradable"]:
        reasons.append("UNKNOWN")
    return [reason for reason in REJECTION_ORDER if reason in reasons]


def _crash_coverage_rows(market: pd.DataFrame, equity: pd.DataFrame, positions: list[PositionWindow]) -> pd.DataFrame:
    mk = market.copy()
    mk["date"] = pd.to_datetime(mk["date"], errors="coerce")
    eq = equity.copy()
    if not eq.empty and "date" in eq:
        eq["date"] = pd.to_datetime(eq["date"], errors="coerce")
    rows: list[dict[str, Any]] = []
    for label, (start_text, end_text) in CRASH_WINDOWS.items():
        start = pd.Timestamp(start_text)
        end = pd.Timestamp(end_text)
        market_window = mk[(mk["date"] >= start) & (mk["date"] <= end)]
        active = [pos for pos in positions if pos.entry_date <= end and pos.exit_date >= start]
        covered_days = _covered_days(active, start, end, market_window["date"] if not market_window.empty else pd.Series(dtype="datetime64[ns]"))
        total_days = int(market_window["date"].nunique()) if not market_window.empty else 0
        rows.append(
            {
                "section": "crash_window_coverage",
                "period": label,
                "crash_start": start_text,
                "crash_end": end_text,
                "max_drawdown": _max_drawdown(market_window, "tx_close"),
                "was_put_spread_active": bool(active),
                "active_position_id": ";".join(pos.position_id for pos in active),
                "days_covered": covered_days,
                "coverage_ratio": _safe_ratio(covered_days, total_days),
                "nearest_put_spread_before_crash": _nearest_before(positions, start),
                "nearest_put_spread_after_crash": _nearest_after(positions, end),
            }
        )
    return pd.DataFrame(rows)


def _rejection_summary_rows(entry: pd.DataFrame) -> pd.DataFrame:
    if entry.empty:
        return pd.DataFrame()
    counts = {reason: 0 for reason in REJECTION_ORDER}
    no_entry = entry[~entry["would_pass_entry_checks"].astype(bool)]
    for reasons in no_entry["rejection_reasons"].fillna(""):
        for reason in str(reasons).split(";"):
            if reason:
                counts[reason] = counts.get(reason, 0) + 1
    rows = [
        {
            "section": "rejection_reason_summary",
            "reason": reason,
            "count": int(count),
            "ratio_of_no_entry_days": _safe_ratio(count, len(no_entry)),
        }
        for reason, count in counts.items()
        if count
    ]
    return pd.DataFrame(rows).sort_values("count", ascending=False) if rows else pd.DataFrame()


def _pre_window_rows(entry: pd.DataFrame) -> pd.DataFrame:
    if entry.empty:
        return pd.DataFrame()
    e = entry.copy()
    e["date"] = pd.to_datetime(e["date"], errors="coerce")
    rows: list[dict[str, Any]] = []
    for label, (start_text, _) in CRASH_WINDOWS.items():
        start = pd.Timestamp(start_text)
        pre_start = start - pd.Timedelta(days=180)
        window = e[(e["date"] >= pre_start) & (e["date"] < start)]
        market_days = window[window["market_conditions_met"].astype(bool)]
        rows.append(
            {
                "section": "crash_pre_window_audit",
                "period": label,
                "pre_window_start": str(pre_start.date()),
                "crash_start": start_text,
                "days_total": int(len(window)),
                "market_condition_days": int(len(market_days)),
                "options_availability_block_days": _reason_count(market_days, ["NO_EXPIRY_IN_DTE_RANGE", "NO_LONG_PUT_CANDIDATE", "NO_SHORT_PUT_CANDIDATE"]),
                "quote_quality_block_days": _reason_count(market_days, ["QUOTE_NOT_VALID"]),
                "liquidity_block_days": _reason_count(market_days, ["LOW_VOLUME", "LOW_OPEN_INTEREST"]),
                "budget_block_days": _reason_count(market_days, ["BUDGET_EXCEEDED"]),
            }
        )
    return pd.DataFrame(rows)


def _forced_unfilled_rows(
    trades: pd.DataFrame,
    lifecycle: pd.DataFrame,
    options: pd.DataFrame,
    config: dict,
    put_params: dict,
) -> pd.DataFrame:
    if lifecycle.empty:
        return pd.DataFrame()
    forced = lifecycle[lifecycle.get("issue", pd.Series(dtype=str)).astype(str) == "forced_unfilled_exit"].copy()
    if forced.empty:
        return pd.DataFrame()
    t = trades.copy()
    if not t.empty:
        t["date"] = pd.to_datetime(t["date"], errors="coerce")
    opt = options.copy()
    if not opt.empty:
        opt["date"] = pd.to_datetime(opt["date"], errors="coerce")
        opt["expiry"] = pd.to_datetime(opt["expiry"], errors="coerce")
    rows: list[dict[str, Any]] = []
    for event in forced.itertuples(index=False):
        pid = str(event.position_id)
        entry_date = pd.Timestamp(getattr(event, "entry_date"))
        expiry = pd.Timestamp(getattr(event, "expiry"))
        pos_trades = t[t.get("position_id", pd.Series(dtype=str)).astype(str) == pid] if not t.empty else pd.DataFrame()
        keys = pos_trades[["expiry", "cp", "strike"]].drop_duplicates() if not pos_trades.empty else pd.DataFrame()
        valid_dates = []
        for key in keys.itertuples(index=False):
            match = opt[
                (opt["expiry"] == pd.Timestamp(key.expiry))
                & (opt["cp"].astype(str) == str(key.cp))
                & (np.isclose(opt["strike"], float(key.strike)))
                & (opt["date"] <= expiry)
                & (opt["date"] >= entry_date)
            ].copy()
            if not match.empty:
                valid = match[match["tradable"].astype(bool) & match["date"].notna()]
                if not valid.empty:
                    valid_dates.append(valid["date"].max())
        last_valid = max(valid_dates) if valid_dates else pd.NaT
        exit_dte = int(put_params.get("exit_dte", 14))
        dte_rule_date = expiry - pd.Timedelta(days=exit_dte)
        should_exit = pd.notna(last_valid) and last_valid <= dte_rule_date
        rows.append(
            {
                "section": "forced_unfilled_impact",
                "position_id": pid,
                "entry_date": str(entry_date.date()),
                "expiry": str(expiry.date()),
                "last_day_with_valid_quote_before_expiry": "" if pd.isna(last_valid) else str(last_valid.date()),
                "days_without_valid_exit_quote": "" if pd.isna(last_valid) else int((expiry - last_valid).days),
                "should_have_exited_earlier_by_dte_rule": bool(should_exit),
                "reason_exit_failed": "No current full-leg tradable quote was available before expiry exit could be generated.",
            }
        )
    return pd.DataFrame(rows)


def _markdown(coverage: pd.DataFrame, entry: pd.DataFrame) -> str:
    crash = coverage[coverage["section"] == "crash_window_coverage"] if not coverage.empty else pd.DataFrame()
    rejection = coverage[coverage["section"] == "rejection_reason_summary"] if not coverage.empty else pd.DataFrame()
    pre = coverage[coverage["section"] == "crash_pre_window_audit"] if not coverage.empty else pd.DataFrame()
    forced = coverage[coverage["section"] == "forced_unfilled_impact"] if not coverage.empty else pd.DataFrame()
    lines = [
        "# Put Spread Coverage Audit",
        "",
        "This audit reads generated reports and processed data only. It does not change strategy rules, parameters, fills, or backtest results.",
        "",
        "## Crash Window Coverage",
        "",
    ]
    if crash.empty:
        lines.append("- No crash coverage rows generated.")
    else:
        for row in crash.itertuples(index=False):
            lines.append(
                f"- {row.period}: active={row.was_put_spread_active}, "
                f"position={row.active_position_id or ''}, days_covered={row.days_covered}, "
                f"coverage_ratio={row.coverage_ratio}, max_drawdown={row.max_drawdown}, "
                f"nearest_before={row.nearest_put_spread_before_crash}, nearest_after={row.nearest_put_spread_after_crash}"
            )
    lines.extend(["", "## Top Rejection Reasons", ""])
    if rejection.empty:
        lines.append("- None.")
    else:
        for row in rejection.head(12).itertuples(index=False):
            lines.append(f"- {row.reason}: {row.count} days ({row.ratio_of_no_entry_days})")
    lines.extend(["", "## Crash Pre-Window Blocks", ""])
    if pre.empty:
        lines.append("- None.")
    else:
        for row in pre.itertuples(index=False):
            lines.append(
                f"- {row.period}: market_condition_days={row.market_condition_days}, "
                f"options_blocks={row.options_availability_block_days}, "
                f"quote_blocks={row.quote_quality_block_days}, liquidity_blocks={row.liquidity_block_days}, "
                f"budget_blocks={row.budget_block_days}"
            )
    lines.extend(["", "## Forced Unfilled Impact", ""])
    if forced.empty:
        lines.append("- None.")
    else:
        for row in forced.itertuples(index=False):
            lines.append(
                f"- {row.position_id}: expiry={row.expiry}, last_valid_quote={row.last_day_with_valid_quote_before_expiry}, "
                f"days_without_valid_exit_quote={row.days_without_valid_exit_quote}, "
                f"dte_rule_should_have_exited={row.should_have_exited_earlier_by_dte_rule}"
            )
    lines.extend(
        [
            "",
            "## Required Limitations",
            "",
            "- This audit explains coverage gaps.",
            "- It does not recommend parameter changes.",
            "- Do not optimize entry thresholds based on crash windows.",
            "- VIX proxy rows are flagged as proxy data and must not be treated as real VIX.",
        ]
    )
    return "\n".join(lines) + "\n"


def _position_windows(trades: pd.DataFrame, lifecycle: pd.DataFrame) -> list[PositionWindow]:
    if trades.empty or "position_id" not in trades:
        return []
    t = trades.copy()
    t["date"] = pd.to_datetime(t["date"], errors="coerce")
    t["expiry"] = pd.to_datetime(t["expiry"], errors="coerce")
    event_map = {}
    if not lifecycle.empty and "position_id" in lifecycle:
        event_map = {str(row["position_id"]): row.to_dict() for _, row in lifecycle.iterrows()}
    windows: list[PositionWindow] = []
    for pid, group in t.groupby("position_id"):
        opens = group[group["reason"].astype(str).str.contains("open", case=False, na=False)]
        exits = group[~group.index.isin(opens.index)]
        if opens.empty:
            continue
        entry = opens["date"].min()
        expiry = group["expiry"].dropna().min()
        event = event_map.get(str(pid), {})
        event_exit = pd.to_datetime(event.get("exit_date", pd.NaT), errors="coerce")
        exit_date = event_exit if pd.notna(event_exit) else (exits["date"].max() if not exits.empty else expiry)
        if pd.notna(entry) and pd.notna(expiry) and pd.notna(exit_date):
            windows.append(PositionWindow(str(pid), entry, expiry, exit_date))
    return windows


def _annual_spend_by_date(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["date", "year", "spend"])
    t = trades.copy()
    t["date"] = pd.to_datetime(t["date"], errors="coerce")
    opens = t[(t.get("strategy") == "put_spread") & t["reason"].astype(str).str.contains("open", case=False, na=False)]
    if opens.empty:
        return pd.DataFrame(columns=["date", "year", "spend"])
    spend = opens.groupby(["date"], as_index=False)["cash_flow"].sum()
    spend = spend.rename(columns={"cash_flow": "spend"})
    spend["year"] = spend["date"].dt.year
    spend["spend"] = (-spend["spend"]).clip(lower=0.0)
    return spend


def _spend_used_before(spend: pd.DataFrame, date: pd.Timestamp) -> float:
    if spend.empty:
        return 0.0
    mask = (spend["year"] == date.year) & (spend["date"] < date)
    return float(spend.loc[mask, "spend"].sum())


def _active_positions(positions: list[PositionWindow], date: pd.Timestamp) -> list[str]:
    return [pos.position_id for pos in positions if pos.entry_date <= date <= pos.exit_date]


def _covered_days(positions: list[PositionWindow], start: pd.Timestamp, end: pd.Timestamp, dates: pd.Series) -> int:
    if not positions or dates.empty:
        return 0
    covered = pd.Series(False, index=dates.index)
    for pos in positions:
        covered |= (dates >= max(start, pos.entry_date)) & (dates <= min(end, pos.exit_date))
    return int(covered.sum())


def _nearest_before(positions: list[PositionWindow], start: pd.Timestamp) -> str:
    before = [pos for pos in positions if pos.exit_date < start]
    if not before:
        return ""
    pos = max(before, key=lambda p: p.exit_date)
    return f"{pos.position_id}:{pos.exit_date.date()} ({(start - pos.exit_date).days}d before)"


def _nearest_after(positions: list[PositionWindow], end: pd.Timestamp) -> str:
    after = [pos for pos in positions if pos.entry_date > end]
    if not after:
        return ""
    pos = min(after, key=lambda p: p.entry_date)
    return f"{pos.position_id}:{pos.entry_date.date()} ({(pos.entry_date - end).days}d after)"


def _max_drawdown(frame: pd.DataFrame, column: str) -> float | str:
    if frame.empty or column not in frame:
        return ""
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return ""
    return float((values / values.cummax() - 1.0).min())


def _reason_count(frame: pd.DataFrame, reasons: list[str]) -> int:
    if frame.empty:
        return 0
    pattern = "|".join(reasons)
    return int(frame["rejection_reasons"].fillna("").str.contains(pattern, regex=True).sum())


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _row(section: str, check: str, status: str, detail: str) -> dict[str, Any]:
    return {"section": section, "check": check, "status": status, "detail": detail}


def _bool_not_nan(value: Any) -> bool:
    if pd.isna(value):
        return False
    return bool(value)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"true", "1", "yes", "y"}


def _float_or_blank(value: Any) -> float | str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(value):
        return ""
    return value


def _safe_ratio(numerator: float, denominator: float) -> float | str:
    if denominator == 0 or pd.isna(denominator):
        return ""
    return float(numerator / denominator)
