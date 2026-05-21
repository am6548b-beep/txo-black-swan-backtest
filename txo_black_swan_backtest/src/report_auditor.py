"""Audit generated reports without changing strategy rules or results."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def run_report_audit(report_dir: Path) -> pd.DataFrame:
    """Read generated reports and write report_audit.csv / report_audit.md."""

    report_dir.mkdir(parents=True, exist_ok=True)
    trades = _read(report_dir / "trades.csv")
    equity = _read(report_dir / "equity_curve.csv")
    summary = _read(report_dir / "summary.csv")
    macro_dashboard = _read(report_dir / "macro_dashboard.csv")
    lifecycle_events = _read(report_dir / "position_lifecycle_events.csv")

    rows: list[dict] = []
    rows.extend(_audit_trades(trades))
    lifecycle = _position_lifecycle_audit(trades, equity, lifecycle_events)
    lifecycle.to_csv(report_dir / "position_lifecycle_audit.csv", index=False)
    rows.extend(_audit_lifecycle(lifecycle))
    rows.extend(_audit_strikes(trades))
    rows.extend(_audit_execution_prices(trades))
    rows.extend(_audit_equity(equity, summary))
    rows.extend(_audit_put_spread_behavior(trades, equity))
    rows.extend(_audit_macro(macro_dashboard, trades, equity))
    rows.extend(_audit_mock_limitations())

    audit = pd.DataFrame(rows, columns=["category", "check", "status", "detail"])
    audit.to_csv(report_dir / "report_audit.csv", index=False)
    (report_dir / "report_audit.md").write_text(_audit_markdown(audit), encoding="utf-8")
    return audit


def _audit_trades(trades: pd.DataFrame) -> list[dict]:
    rows = []
    required = {"date", "position_id", "strategy", "action", "cp", "strike", "expiry", "quantity", "price", "cash_flow", "cost", "reason", "dte_at_trade"}
    audit_cols = {
        "underlying_price_at_trade",
        "txf_close_at_trade",
        "option_bid_at_trade",
        "option_ask_at_trade",
        "option_close_at_trade",
        "slippage_pct_used",
        "commission_paid",
        "tax_paid",
        "liquidity_flag",
        "bid_ask_estimated",
        "iv_estimated",
        "delta_estimated",
        "quote_quality_status",
        "spread_pct",
        "is_tradable_quote",
    }
    if trades.empty:
        return [_row("trades_integrity", "trades_csv_present", "FAIL", "trades.csv missing or empty")]
    missing = sorted(required - set(trades.columns))
    rows.append(_row("trades_integrity", "required_columns", "PASS" if not missing else "FAIL", ",".join(missing)))
    missing_audit = sorted(audit_cols - set(trades.columns))
    rows.append(_row("trades_integrity", "audit_columns_present", "PASS" if not missing_audit else "WARN", ",".join(missing_audit)))
    if missing:
        return rows

    rows.append(_finite_check("trades_integrity", "price_positive", trades["price"], positive=True))
    rows.append(_finite_check("trades_integrity", "abs_qty_positive", trades["quantity"].abs(), positive=True))
    rows.append(_finite_check("trades_integrity", "cash_flow_finite", trades["cash_flow"]))
    rows.append(_finite_check("trades_integrity", "cost_finite", trades["cost"]))
    rows.extend(_audit_trade_dates(trades))

    ps = trades[trades["strategy"] == "put_spread"].copy()
    if ps.empty:
        rows.append(_row("trades_integrity", "put_spread_trades_present", "WARN", "no put_spread trades in report"))
        return rows

    open_rows = ps[ps["reason"].str.contains("open", case=False, na=False)]
    close_rows = ps[~ps.index.isin(open_rows.index)]
    rows.append(_row("trades_integrity", "put_spread_open_paired", "PASS" if _paired_put_spread(open_rows, opening=True) else "FAIL", "open groups must contain long put BUY and short put SELL"))
    rows.append(_row("trades_integrity", "put_spread_close_paired", "PASS" if close_rows.empty or _paired_put_spread(close_rows, opening=False) else "FAIL", "close groups must reverse both legs"))

    leg_balance = ps.groupby(["position_id", "cp", "strike", "expiry"])["quantity"].sum()
    orphan = leg_balance[leg_balance != 0]
    rows.append(_row("trades_integrity", "orphan_or_open_legs", "PASS" if orphan.empty else "WARN", f"unbalanced legs={len(orphan)}"))
    net_by_position = ps.groupby("position_id")["quantity"].sum()
    rows.append(_row("trades_integrity", "single_leg_unclosed", "PASS" if (net_by_position == 0).all() else "WARN", f"positions with net signed qty={int((net_by_position != 0).sum())}"))
    return rows


def _audit_trade_dates(trades: pd.DataFrame) -> list[dict]:
    if "dte_at_trade" not in trades:
        return [_row("trade_dates", "missing_dte_at_trade", "FAIL", "dte_at_trade column missing")]
    rows: list[dict] = []
    dte = pd.to_numeric(trades["dte_at_trade"], errors="coerce")
    missing = int(dte.isna().sum())
    rows.append(_row("trade_dates", "missing_dte_at_trade", "FAIL" if missing else "PASS", f"rows={missing}"))
    negative = int((dte < 0).sum())
    rows.append(_row("trade_dates", "dte_at_trade_negative", "FAIL" if negative else "PASS", f"rows={negative}"))
    trade_date = pd.to_datetime(trades["date"], errors="coerce")
    expiry = pd.to_datetime(trades["expiry"], errors="coerce")
    after_expiry = int((trade_date > expiry).sum())
    rows.append(_row("trade_dates", "trade_date_after_expiry", "FAIL" if after_expiry else "PASS", f"rows={after_expiry}"))
    exit_rows = trades[~trades["reason"].str.contains("open", case=False, na=False)].copy()
    exit_after = 0 if exit_rows.empty else int((pd.to_datetime(exit_rows["date"], errors="coerce") > pd.to_datetime(exit_rows["expiry"], errors="coerce")).sum())
    rows.append(_row("trade_dates", "exit_after_expiry", "FAIL" if exit_after else "PASS", f"rows={exit_after}"))
    expected = (expiry - trade_date).dt.days
    mismatch = int((dte.notna() & expected.notna() & (dte != expected)).sum())
    rows.append(_row("trade_dates", "quote_lookup_mismatch_expiry", "FAIL" if mismatch else "PASS", f"dte_mismatch_rows={mismatch}"))
    return rows


def _position_lifecycle_audit(trades: pd.DataFrame, equity: pd.DataFrame, lifecycle_events: pd.DataFrame | None = None) -> pd.DataFrame:
    columns = [
        "position_id",
        "entry_date",
        "expiry",
        "exit_date",
        "entry_legs",
        "exit_legs",
        "min_dte_at_exit",
        "max_dte_at_exit",
        "exit_reason",
        "status",
        "issue",
    ]
    if trades.empty or "position_id" not in trades:
        return pd.DataFrame(columns=columns)
    last_date = pd.to_datetime(equity["date"], errors="coerce").max() if not equity.empty and "date" in equity else pd.NaT
    event_map = {}
    if lifecycle_events is not None and not lifecycle_events.empty and "position_id" in lifecycle_events:
        event_map = {str(row.position_id): row for row in lifecycle_events.itertuples(index=False)}
    records: list[dict] = []
    for pid, group in trades.groupby("position_id"):
        open_rows = group[group["reason"].str.contains("open", case=False, na=False)].copy()
        exit_rows = group[~group.index.isin(open_rows.index)].copy()
        entry_date = pd.to_datetime(open_rows["date"], errors="coerce").min() if not open_rows.empty else pd.NaT
        expiry = pd.to_datetime(group["expiry"], errors="coerce").min()
        exit_date = pd.to_datetime(exit_rows["date"], errors="coerce").max() if not exit_rows.empty else pd.NaT
        exit_dte = pd.to_numeric(exit_rows["dte_at_trade"], errors="coerce") if "dte_at_trade" in exit_rows else pd.Series(dtype=float)
        duplicate_exit = False
        if not exit_rows.empty:
            duplicate_exit = bool(exit_rows.duplicated(["position_id", "cp", "strike", "expiry", "action", "reason", "date"], keep=False).any())
        event = event_map.get(str(pid))
        if event is not None:
            status = str(getattr(event, "status", "WARN"))
            issue = str(getattr(event, "issue", "forced_unfilled_exit"))
            event_exit = pd.to_datetime(getattr(event, "exit_date", pd.NaT), errors="coerce")
            if pd.notna(event_exit):
                exit_date = event_exit
        elif not exit_rows.empty and pd.notna(exit_date) and pd.notna(expiry) and exit_date > expiry:
            status = "FAIL"
            issue = "exit_after_expiry"
        elif duplicate_exit:
            status = "FAIL"
            issue = "duplicate_exit_legs"
        elif exit_rows.empty and pd.notna(last_date) and pd.notna(expiry) and last_date > expiry:
            status = "FAIL"
            issue = "expired_position_still_active"
        elif exit_rows.empty:
            status = "WARN"
            issue = "open_position_no_exit"
        else:
            status = "PASS"
            issue = ""
        records.append(
            {
                "position_id": pid,
                "entry_date": "" if pd.isna(entry_date) else str(entry_date.date()),
                "expiry": "" if pd.isna(expiry) else str(expiry.date()),
                "exit_date": "" if pd.isna(exit_date) else str(exit_date.date()),
                "entry_legs": int(len(open_rows)),
                "exit_legs": int(len(exit_rows)),
                "min_dte_at_exit": "" if exit_dte.empty or exit_dte.isna().all() else int(exit_dte.min()),
                "max_dte_at_exit": "" if exit_dte.empty or exit_dte.isna().all() else int(exit_dte.max()),
                "exit_reason": ";".join(sorted(exit_rows["reason"].dropna().astype(str).unique())) if not exit_rows.empty else "",
                "status": status,
                "issue": issue,
            }
        )
    return pd.DataFrame(records, columns=columns)


def _audit_lifecycle(lifecycle: pd.DataFrame) -> list[dict]:
    if lifecycle.empty:
        return [_row("position_lifecycle", "position_lifecycle_audit_present", "WARN", "no positions")]
    rows = [_row("position_lifecycle", "position_lifecycle_audit_present", "PASS", f"rows={len(lifecycle)}")]
    counts = lifecycle["status"].value_counts().to_dict()
    aggregate_status = "FAIL" if counts.get("FAIL", 0) else ("WARN" if counts.get("WARN", 0) else "PASS")
    rows.append(_row("position_lifecycle", "position_lifecycle_status_distribution", aggregate_status, str(counts)))
    for issue in ["expired_position_still_active", "duplicate_exit_legs", "exit_after_expiry"]:
        count = int((lifecycle["issue"] == issue).sum())
        rows.append(_row("position_lifecycle", issue, "FAIL" if count else "PASS", f"rows={count}"))
    return rows


def _audit_strikes(trades: pd.DataFrame) -> list[dict]:
    if trades.empty or "strategy" not in trades:
        return [_row("strike_reasonableness", "strike_ratio_check", "WARN", "trades unavailable")]
    ps_open = trades[(trades["strategy"] == "put_spread") & (trades["reason"].str.contains("open", case=False, na=False))]
    if ps_open.empty:
        return [_row("strike_reasonableness", "strike_ratio_check", "WARN", "no put spread opens")]
    rows = []
    for pid, group in ps_open.groupby("position_id"):
        buys = group[(group["action"] == "BUY") & (group["cp"] == "P")]
        sells = group[(group["action"] == "SELL") & (group["cp"] == "P")]
        if buys.empty or sells.empty:
            rows.append(_row("strike_reasonableness", "put_spread_strike_pair", "FAIL", f"{pid}: missing long or short put"))
            continue
        long_strike = float(buys["strike"].max())
        short_strike = float(sells["strike"].min())
        if "txf_close_at_trade" not in group or group["txf_close_at_trade"].isna().all():
            implied_underlying = long_strike / 0.90
            long_ratio = long_strike / implied_underlying
            short_ratio = short_strike / implied_underlying
            rows.append(_row("strike_reasonableness", "txf_close_available", "WARN", f"{pid}: txf_close_at_trade missing; using proxy"))
        else:
            txf = float(group["txf_close_at_trade"].dropna().iloc[0])
            long_ratio = long_strike / txf
            short_ratio = short_strike / txf
        long_ok = min(abs(long_ratio - 0.90), abs(long_ratio - 0.93)) <= 0.04
        short_ok = min(abs(short_ratio - 0.75), abs(short_ratio - 0.78)) <= 0.06
        status = "PASS" if long_ok and short_ok else "WARN"
        detail = f"{pid}: long_ratio={long_ratio:.3f}, short_ratio={short_ratio:.3f}"
        rows.append(_row("strike_reasonableness", "put_spread_moneyness", status, detail))
    return rows


def _audit_execution_prices(trades: pd.DataFrame) -> list[dict]:
    rows = []
    if trades.empty:
        return [_row("execution_reasonableness", "execution_report_available", "FAIL", "trades unavailable")]
    buy_ok = (trades.loc[trades["action"] == "BUY", "cash_flow"] < 0).all()
    sell_ok = (trades.loc[trades["action"] == "SELL", "cash_flow"] > 0).all()
    rows.append(_row("execution_reasonableness", "buy_cash_flow_negative", "PASS" if buy_ok else "FAIL", "BUY should consume cash"))
    rows.append(_row("execution_reasonableness", "sell_cash_flow_positive", "PASS" if sell_ok else "FAIL", "SELL should add cash after costs"))
    needed = {"option_bid_at_trade", "option_ask_at_trade", "option_close_at_trade", "slippage_pct_used"}
    if needed <= set(trades.columns):
        expected = []
        for row in trades.itertuples(index=False):
            slip = float(getattr(row, "slippage_pct_used"))
            if row.action == "BUY":
                expected.append(float(row.option_ask_at_trade) * (1.0 + slip))
            elif row.action == "SELL":
                expected.append(max(0.0, float(row.option_bid_at_trade) * (1.0 - slip)))
            else:
                expected.append(np.nan)
        diff = (pd.Series(expected) - pd.to_numeric(trades["price"], errors="coerce")).abs()
        rows.append(_row("execution_reasonableness", "bid_ask_slippage_fill_price", "PASS" if diff.max() < 1e-8 else "FAIL", f"max_abs_diff={diff.max():.10f}"))
        close_fill = (pd.to_numeric(trades["price"], errors="coerce") - pd.to_numeric(trades["option_close_at_trade"], errors="coerce")).abs() < 1e-10
        rows.append(_row("execution_reasonableness", "close_not_used_as_universal_fill", "PASS" if not close_fill.all() else "FAIL", f"close_equal_fill_legs={int(close_fill.sum())}"))
    else:
        rows.append(_row("execution_reasonableness", "bid_ask_slippage_fill_price", "WARN", f"missing columns={sorted(needed - set(trades.columns))}"))
        rows.append(_row("execution_reasonableness", "close_not_used_as_universal_fill", "WARN", "option_close_at_trade unavailable"))
    if "bid_ask_estimated" in trades:
        estimated = _bool_series(trades["bid_ask_estimated"])
        rows.append(_row("execution_reasonableness", "bid_ask_estimated_flag", "WARN" if estimated.any() else "PASS", f"estimated_legs={int(estimated.sum())}"))
    else:
        rows.append(_row("execution_reasonableness", "bid_ask_estimated_flag", "WARN", "bid_ask_estimated column missing"))
    for col in ["iv_estimated", "delta_estimated"]:
        if col in trades:
            estimated = _bool_series(trades[col])
            rows.append(_row("execution_reasonableness", col, "WARN" if estimated.any() else "PASS", f"estimated_legs={int(estimated.sum())}"))
    if {"commission_paid", "tax_paid"} <= set(trades.columns):
        cost_diff = (pd.to_numeric(trades["commission_paid"], errors="coerce") + pd.to_numeric(trades["tax_paid"], errors="coerce") - pd.to_numeric(trades["cost"], errors="coerce")).abs().max()
        rows.append(_row("execution_reasonableness", "commission_tax_sum_to_cost", "PASS" if cost_diff < 1e-8 else "FAIL", f"max_abs_diff={cost_diff:.10f}"))
    rows.extend(_audit_trade_quote_quality(trades))
    total_cost = float(trades["cost"].sum()) if "cost" in trades else 0.0
    turnover = float(trades["cash_flow"].abs().sum()) if "cash_flow" in trades else 0.0
    rows.append(_row("execution_reasonableness", "fee_tax_cost_summary", "PASS", f"reported_cost={total_cost:.2f}, turnover={turnover:.2f}; slippage_pct_used is now recorded per leg"))
    return rows


def _audit_trade_quote_quality(trades: pd.DataFrame) -> list[dict]:
    needed = {"quote_quality_status", "is_tradable_quote"}
    if not needed <= set(trades.columns):
        return [
            _row(
                "quote_quality",
                "quote_quality_columns_present",
                "WARN",
                f"missing columns={sorted(needed - set(trades.columns))}",
            )
        ]
    status = trades["quote_quality_status"].fillna("UNKNOWN").astype(str)
    tradable = _bool_series(trades["is_tradable_quote"])
    non_valid = status.ne("VALID")
    non_tradable = ~tradable
    distribution = "; ".join(f"{idx}={val}" for idx, val in status.value_counts(dropna=False).items())
    return [
        _row("quote_quality", "traded_legs_quote_quality_status_distribution", "PASS", distribution),
        _row(
            "quote_quality",
            "no_trades_with_is_tradable_quote_false",
            "FAIL" if non_tradable.any() else "PASS",
            f"bad_legs={int(non_tradable.sum())}",
        ),
        _row(
            "quote_quality",
            "no_trades_with_non_valid_quote_quality",
            "FAIL" if non_valid.any() else "PASS",
            f"bad_legs={int(non_valid.sum())}",
        ),
    ]


def _audit_equity(equity: pd.DataFrame, summary: pd.DataFrame) -> list[dict]:
    if equity.empty:
        return [_row("equity_curve", "equity_curve_present", "FAIL", "equity_curve.csv missing or empty")]
    rows = []
    numeric_cols = ["cash", "stock_equity", "option_value", "total_equity"]
    for col in numeric_cols:
        if col not in equity:
            rows.append(_row("equity_curve", f"{col}_present", "FAIL", "missing column"))
            continue
        rows.append(_finite_check("equity_curve", f"{col}_finite", equity[col]))
    rows.append(_row("equity_curve", "total_equity_positive", "PASS" if (equity["total_equity"] > 0).all() else "FAIL", "total_equity must not be negative or zero"))
    dd = equity["total_equity"] / equity["total_equity"].cummax() - 1.0
    rows.append(_row("equity_curve", "drawdown_non_positive", "PASS" if (dd <= 1e-12).all() else "FAIL", f"max_drawdown_curve_value={dd.max():.8f}"))
    if not summary.empty and "max_drawdown" in summary:
        reported = float(summary["max_drawdown"].iloc[0])
        calc = float(dd.min())
        rows.append(_row("equity_curve", "summary_max_drawdown_matches", "PASS" if abs(reported - calc) < 1e-8 else "FAIL", f"summary={reported:.10f}, equity_curve={calc:.10f}"))
    calc_total = equity["cash"] + equity["stock_equity"] + equity["option_value"]
    diff = (calc_total - equity["total_equity"]).abs().max()
    rows.append(_row("equity_curve", "accounting_identity", "PASS" if diff < 1e-6 else "FAIL", f"max_abs_diff={diff:.6f}"))
    daily = equity["total_equity"].pct_change().abs()
    jumps = int((daily > 0.10).sum())
    rows.append(_row("equity_curve", "single_day_move_over_10pct", "WARN" if jumps else "PASS", f"days={jumps}"))
    return rows


def _audit_put_spread_behavior(trades: pd.DataFrame, equity: pd.DataFrame) -> list[dict]:
    rows = []
    if trades.empty or equity.empty:
        return [_row("put_spread_behavior", "inputs_available", "WARN", "trades or equity unavailable")]
    if "option_value" not in equity or "stock_equity" not in equity:
        return [_row("put_spread_behavior", "mtm_columns_available", "WARN", "option_value or stock_equity missing")]
    stock_dd = equity["stock_equity"] / equity["stock_equity"].cummax() - 1.0
    crash = stock_dd <= -0.10
    if crash.any():
        first_crash_idx = crash.idxmax()
        window = equity.loc[first_crash_idx : min(first_crash_idx + 10, len(equity) - 1)]
        option_helped = (window["option_value"].max() > equity.loc[max(first_crash_idx - 1, 0), "option_value"]) or (window["daily_option_pnl"].max() > 0 if "daily_option_pnl" in window else False)
        rows.append(_row("put_spread_behavior", "option_mtm_rises_on_market_drop", "PASS" if option_helped else "WARN", "uses stock_equity drawdown proxy because tx_close is not in equity report"))
    else:
        rows.append(_row("put_spread_behavior", "market_drop_observed", "WARN", "no stock_equity drawdown >= 10% in report"))
    ps = trades[trades["strategy"] == "put_spread"] if "strategy" in trades else pd.DataFrame()
    if ps.empty:
        rows.append(_row("put_spread_behavior", "put_spread_exits", "WARN", "no put spread trades"))
        return rows
    ps = ps.copy()
    ps["date"] = pd.to_datetime(ps["date"])
    ps["expiry"] = pd.to_datetime(ps["expiry"])
    late = ps[(~ps["reason"].str.contains("open", case=False, na=False)) & (ps["date"] >= ps["expiry"])]
    rows.append(_row("put_spread_behavior", "exits_before_expiry", "PASS" if late.empty else "FAIL", f"late_exit_legs={len(late)}"))
    return rows


def _audit_macro(macro_dashboard: pd.DataFrame, trades: pd.DataFrame, equity: pd.DataFrame) -> list[dict]:
    if macro_dashboard.empty:
        return [_row("macro_risk", "macro_dashboard_present", "WARN", "macro_dashboard.csv missing")]
    rows = [_row("macro_risk", "macro_dashboard_present", "PASS", "")]
    status_map = dict(zip(macro_dashboard["item"], macro_dashboard["status"]))
    value_map = dict(zip(macro_dashboard["item"], macro_dashboard["value"]))
    red_or_orange = macro_dashboard[macro_dashboard["status"].isin(["RED", "ORANGE"])]
    rows.append(_row("macro_risk", "dashboard_red_or_orange_flags", "WARN" if not red_or_orange.empty else "PASS", f"count={len(red_or_orange)}"))
    rows.append(_row("macro_risk", "revenue_growth_gt_unit_growth", _status_from_item(status_map, "ASP growth vs unit growth"), f"value={value_map.get('ASP growth vs unit growth', 'NA')}"))
    rows.append(_row("macro_risk", "inventory_days_rising_component_risk", _status_from_item(status_map, "Component inventory days"), f"value={value_map.get('Component inventory days', 'NA')}"))
    upstream_extreme = status_map.get("AI server capex growth") in {"RED", "ORANGE"} and status_map.get("Consumer sentiment") in {"RED", "ORANGE"}
    rows.append(_row("macro_risk", "consumer_weak_but_upstream_extreme", "WARN" if upstream_extreme else "PASS", f"consumer={status_map.get('Consumer sentiment')}, capex={status_map.get('AI server capex growth')}"))
    macro_states = set(equity["macro_state"]) if not equity.empty and "macro_state" in equity else set()
    rows.append(_row("macro_risk", "ai_supply_distortion_marked", "PASS" if "AI_SUPPLY_DISTORTION" in macro_states else "WARN", f"states={sorted(macro_states)}"))
    ic_open = trades[(trades["strategy"] == "iron_condor") & (trades["reason"].str.contains("open", case=False, na=False))] if not trades.empty and "strategy" in trades else pd.DataFrame()
    collapse_states = {"BULLWHIP_COLLAPSE", "STAGFLATION_DEMAND_BREAK"}
    if macro_states & collapse_states:
        status = "PASS" if ic_open.empty else "FAIL"
        detail = f"collapse_states={sorted(macro_states & collapse_states)}, ic_open_legs={len(ic_open)}"
    else:
        status = "WARN"
        detail = "no BULLWHIP_COLLAPSE/STAGFLATION_DEMAND_BREAK state in equity report"
    rows.append(_row("macro_risk", "bullwhip_collapse_blocks_iron_condor", status, detail))
    return rows


def _audit_mock_limitations() -> list[dict]:
    limitations = [
        "mock data may have overly smooth bid/ask",
        "volume/open_interest may be too idealized",
        "VIX may be overly smooth",
        "real night-session gap risk is missing",
        "real volatility skew is missing",
        "mock results must not be used for investment decisions",
    ]
    return [_row("mock_data_limitation", item, "WARN", item) for item in limitations]


def _paired_put_spread(frame: pd.DataFrame, opening: bool) -> bool:
    if frame.empty:
        return False
    for _, group in frame.groupby(["position_id", "date", "reason"]):
        buys = group[(group["action"] == ("BUY" if opening else "BUY")) & (group["cp"] == "P")]
        sells = group[(group["action"] == ("SELL" if opening else "SELL")) & (group["cp"] == "P")]
        if opening:
            if not ((group["action"] == "BUY").any() and (group["action"] == "SELL").any()):
                return False
        else:
            if not ((group["action"] == "BUY").any() and (group["action"] == "SELL").any()):
                return False
        if buys.empty or sells.empty:
            return False
    return True


def _finite_check(category: str, check: str, series: pd.Series, positive: bool = False) -> dict:
    values = pd.to_numeric(series, errors="coerce")
    finite = np.isfinite(values).all()
    if positive:
        finite = finite and (values > 0).all()
    return _row(category, check, "PASS" if finite else "FAIL", "")


def _bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin({"true", "1", "yes", "y"})


def _status_from_item(status_map: dict, item: str) -> str:
    status = status_map.get(item)
    if status in {"RED", "ORANGE"}:
        return "WARN"
    if status in {"GREEN", "YELLOW"}:
        return "PASS"
    return "WARN"


def _read(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _row(category: str, check: str, status: str, detail: str) -> dict:
    return {"category": category, "check": check, "status": status, "detail": detail}


def _audit_markdown(audit: pd.DataFrame) -> str:
    counts = audit["status"].value_counts().to_dict() if not audit.empty else {}
    fail = audit[audit["status"] == "FAIL"]
    warn = audit[audit["status"] == "WARN"]
    lines = [
        "# Report Audit",
        "",
        "This audit reads generated reports only. It does not change strategy rules, parameters, fills, or backtest results.",
        "",
        "## Status Counts",
        "",
        f"- PASS: {counts.get('PASS', 0)}",
        f"- WARN: {counts.get('WARN', 0)}",
        f"- FAIL: {counts.get('FAIL', 0)}",
        "",
        "## Major FAIL",
        "",
    ]
    if fail.empty:
        lines.append("- None")
    else:
        for row in fail.itertuples(index=False):
            lines.append(f"- `{row.category}.{row.check}`: {row.detail}")
    lines.extend(["", "## Key WARN", ""])
    if warn.empty:
        lines.append("- None")
    else:
        for row in warn.head(12).itertuples(index=False):
            lines.append(f"- `{row.category}.{row.check}`: {row.detail}")
    lines.extend(
        [
            "",
            "## Mock Data Limitations",
            "",
            "- mock data may have overly smooth bid/ask",
            "- volume/open_interest may be too idealized",
            "- VIX may be overly smooth",
            "- real night-session gap risk is missing",
            "- real volatility skew is missing",
            "- mock results must not be used for investment decisions",
        ]
    )
    return "\n".join(lines) + "\n"
