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

    rows: list[dict] = []
    rows.extend(_audit_trades(trades))
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
    required = {"date", "position_id", "strategy", "action", "cp", "strike", "expiry", "quantity", "price", "cash_flow", "cost", "reason"}
    if trades.empty:
        return [_row("trades_integrity", "trades_csv_present", "FAIL", "trades.csv missing or empty")]
    missing = sorted(required - set(trades.columns))
    rows.append(_row("trades_integrity", "required_columns", "PASS" if not missing else "FAIL", ",".join(missing)))
    if missing:
        return rows

    rows.append(_finite_check("trades_integrity", "price_positive", trades["price"], positive=True))
    rows.append(_finite_check("trades_integrity", "abs_qty_positive", trades["quantity"].abs(), positive=True))
    rows.append(_finite_check("trades_integrity", "cash_flow_finite", trades["cash_flow"]))
    rows.append(_finite_check("trades_integrity", "cost_finite", trades["cost"]))

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
        implied_underlying = long_strike / 0.90
        long_ratio = long_strike / implied_underlying
        short_ratio = short_strike / implied_underlying
        long_ok = min(abs(long_ratio - 0.90), abs(long_ratio - 0.93)) <= 0.04
        short_ok = min(abs(short_ratio - 0.75), abs(short_ratio - 0.78)) <= 0.06
        status = "PASS" if long_ok and short_ok else "WARN"
        detail = f"{pid}: report lacks txf_close; using long_put/0.90 implied reference. long_ratio={long_ratio:.3f}, short_ratio={short_ratio:.3f}"
        rows.append(_row("strike_reasonableness", "put_spread_moneyness_proxy", status, detail))
    rows.append(_row("strike_reasonableness", "txf_close_available", "WARN", "reports do not include txf_close, exact moneyness cannot be verified from report-only inputs"))
    return rows


def _audit_execution_prices(trades: pd.DataFrame) -> list[dict]:
    rows = []
    if trades.empty:
        return [_row("execution_reasonableness", "execution_report_available", "FAIL", "trades unavailable")]
    buy_ok = (trades.loc[trades["action"] == "BUY", "cash_flow"] < 0).all()
    sell_ok = (trades.loc[trades["action"] == "SELL", "cash_flow"] > 0).all()
    rows.append(_row("execution_reasonableness", "buy_cash_flow_negative", "PASS" if buy_ok else "FAIL", "BUY should consume cash"))
    rows.append(_row("execution_reasonableness", "sell_cash_flow_positive", "PASS" if sell_ok else "FAIL", "SELL should add cash after costs"))
    rows.append(_row("execution_reasonableness", "bid_ask_slippage_verifiable", "WARN", "trades.csv has fill price only; bid/ask/close are not present, so audit cannot prove ask+slippage or bid-slippage from reports alone"))
    rows.append(_row("execution_reasonableness", "close_not_used_as_universal_fill", "WARN", "close column is absent from trades.csv; no evidence of close fills in report, but raw quote audit is required for proof"))
    total_cost = float(trades["cost"].sum()) if "cost" in trades else 0.0
    turnover = float(trades["cash_flow"].abs().sum()) if "cash_flow" in trades else 0.0
    rows.append(_row("execution_reasonableness", "fee_tax_cost_summary", "PASS", f"reported_cost={total_cost:.2f}, turnover={turnover:.2f}; slippage is embedded in fill price and cannot be separated without quote snapshot"))
    return rows


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
