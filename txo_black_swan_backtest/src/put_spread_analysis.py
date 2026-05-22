"""Diagnostic analysis for real-data put-spread backtest reports.

This module reads generated reports and processed data. It does not change
strategy rules, parameters, fills, or backtest results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CRASH_WINDOWS: dict[str, tuple[str, str]] = {
    "2008": ("2008-01-01", "2008-12-31"),
    "2011": ("2011-01-01", "2011-12-31"),
    "2015": ("2015-06-01", "2015-12-31"),
    "2018": ("2018-01-01", "2018-12-31"),
    "2020": ("2020-01-01", "2020-12-31"),
    "2022": ("2022-01-01", "2022-12-31"),
}


def write_put_spread_real_data_analysis(report_dir: Path, processed_data_dir: Path) -> pd.DataFrame:
    """Write put_spread_real_data_analysis.csv and .md.

    The output is diagnostic only. It intentionally avoids performance
    recommendations or parameter selection.
    """

    report_dir.mkdir(parents=True, exist_ok=True)
    trades = _read_csv(report_dir / "trades.csv")
    equity = _read_csv(report_dir / "equity_curve.csv")
    summary = _read_csv(report_dir / "summary.csv")
    report_audit = _read_csv(report_dir / "report_audit.csv")
    lifecycle = _read_csv(report_dir / "position_lifecycle_audit.csv")
    data_quality = _read_csv(report_dir / "data_quality_report.csv")
    quote_quality = _read_csv(report_dir / "options_quote_quality.csv")
    market = _read_market(processed_data_dir / "market.csv")
    option_dates = _read_option_dates(processed_data_dir / "options.csv")

    frames = [
        _data_basis_rows(market, option_dates, data_quality, trades, lifecycle),
        _trade_quality_rows(trades),
        _position_rows(trades, equity, lifecycle, market),
        _annual_hedge_cost_rows(trades, equity),
        _crash_effectiveness_rows(trades, equity, market),
        _forced_exit_rows(trades, lifecycle, processed_data_dir),
        _audit_context_rows(summary, report_audit, data_quality, quote_quality),
    ]
    analysis = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True)
    analysis.to_csv(report_dir / "put_spread_real_data_analysis.csv", index=False)
    (report_dir / "put_spread_real_data_analysis.md").write_text(
        _analysis_markdown(analysis),
        encoding="utf-8",
    )
    return analysis


def _data_basis_rows(
    market: pd.DataFrame,
    option_dates: pd.DataFrame,
    data_quality: pd.DataFrame,
    trades: pd.DataFrame,
    lifecycle: pd.DataFrame,
) -> pd.DataFrame:
    market_rows = _row_count_from_quality(data_quality, "market.csv")
    option_rows = _row_count_from_quality(data_quality, "options.csv")
    forced_count = int((lifecycle.get("issue", pd.Series(dtype=str)).astype(str) == "forced_unfilled_exit").sum()) if not lifecycle.empty else 0
    rows = [
        {
            "section": "data_basis",
            "metric": "market_date_range",
            "value": _date_range(market, "date"),
        },
        {
            "section": "data_basis",
            "metric": "options_date_range",
            "value": _date_range(option_dates, "date"),
        },
        {"section": "data_basis", "metric": "market_rows", "value": market_rows},
        {"section": "data_basis", "metric": "options_rows", "value": option_rows},
        {"section": "data_basis", "metric": "trades_count", "value": int(len(trades))},
        {"section": "data_basis", "metric": "position_count", "value": _position_count(trades, lifecycle)},
        {"section": "data_basis", "metric": "forced_unfilled_exit_count", "value": forced_count},
    ]
    return pd.DataFrame(rows)


def _trade_quality_rows(trades: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if trades.empty:
        return pd.DataFrame(
            [
                {
                    "section": "trade_quality",
                    "metric": "trades_available",
                    "status": "WARN",
                    "value": "no trades.csv rows",
                }
            ]
        )
    status = trades.get("quote_quality_status", pd.Series(["MISSING_COLUMN"] * len(trades))).fillna("UNKNOWN").astype(str)
    for item, count in status.value_counts(dropna=False).items():
        rows.append(
            {
                "section": "trade_quote_quality_distribution",
                "quote_quality_status": item,
                "count": int(count),
            }
        )
    tradable = _bool_series(trades.get("is_tradable_quote", pd.Series([False] * len(trades))))
    spread = pd.to_numeric(trades.get("spread_pct", pd.Series(dtype=float)), errors="coerce")
    rows.extend(
        [
            {"section": "trade_quality", "metric": "is_tradable_quote_false_count", "value": int((~tradable).sum())},
            {"section": "trade_quality", "metric": "average_spread_pct_traded_legs", "value": _float_or_blank(spread.mean())},
            {"section": "trade_quality", "metric": "max_spread_pct_traded_legs", "value": _float_or_blank(spread.max())},
            {"section": "trade_quality", "metric": "bid_ask_estimated_count", "value": int(_bool_series(trades.get("bid_ask_estimated", pd.Series(dtype=bool))).sum())},
            {"section": "trade_quality", "metric": "iv_estimated_count", "value": int(_bool_series(trades.get("iv_estimated", pd.Series(dtype=bool))).sum())},
            {"section": "trade_quality", "metric": "delta_estimated_count", "value": int(_bool_series(trades.get("delta_estimated", pd.Series(dtype=bool))).sum())},
        ]
    )
    return pd.DataFrame(rows)


def _position_rows(trades: pd.DataFrame, equity: pd.DataFrame, lifecycle: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    if trades.empty or "position_id" not in trades:
        return pd.DataFrame()
    t = trades.copy()
    t["date"] = pd.to_datetime(t["date"], errors="coerce")
    t["expiry"] = pd.to_datetime(t["expiry"], errors="coerce")
    eq = _prepare_equity(equity)
    mk = _prepare_market(market)
    lifecycle_by_id = _lifecycle_map(lifecycle)
    rows: list[dict[str, Any]] = []
    for pid, group in t.groupby("position_id"):
        opens = group[group["reason"].astype(str).str.contains("open", case=False, na=False)].copy()
        exits = group[~group.index.isin(opens.index)].copy()
        if opens.empty:
            continue
        entry_date = opens["date"].min()
        expiry = group["expiry"].dropna().min()
        event = lifecycle_by_id.get(str(pid), {})
        event_exit = pd.to_datetime(event.get("exit_date", pd.NaT), errors="coerce")
        trade_exit = exits["date"].max() if not exits.empty else pd.NaT
        exit_date = event_exit if pd.notna(event_exit) else trade_exit
        if pd.isna(exit_date):
            exit_date = expiry
        entry_underlying = _first_numeric(opens, ["txf_close_at_trade", "underlying_price_at_trade"])
        long_put = opens[(opens["action"] == "BUY") & (opens["cp"] == "P")]
        short_put = opens[(opens["action"] == "SELL") & (opens["cp"] == "P")]
        long_strike = float(long_put["strike"].max()) if not long_put.empty else np.nan
        short_strike = float(short_put["strike"].min()) if not short_put.empty else np.nan
        eq_window = _window(eq, entry_date, exit_date)
        mk_window = _window(mk, entry_date, exit_date)
        rows.append(
            {
                "section": "position",
                "position_id": pid,
                "entry_date": _date_text(entry_date),
                "expiry": _date_text(expiry),
                "entry_underlying": _float_or_blank(entry_underlying),
                "long_put_strike": _float_or_blank(long_strike),
                "short_put_strike": _float_or_blank(short_strike),
                "long_put_moneyness": _safe_ratio(long_strike, entry_underlying),
                "short_put_moneyness": _safe_ratio(short_strike, entry_underlying),
                "net_premium_paid": _float_or_blank(-opens["cash_flow"].sum()),
                "exit_date": _date_text(exit_date),
                "exit_reason": _exit_reason(exits, event),
                "realized_pnl": _float_or_blank(group["cash_flow"].sum()),
                "forced_unfilled_exit": bool(event.get("issue") == "forced_unfilled_exit"),
                "max_drawdown_during_position": _window_drawdown(eq_window),
                "underlying_return_during_position": _underlying_return(mk_window),
                "option_mtm_max": _float_or_blank(pd.to_numeric(eq_window.get("option_value", pd.Series(dtype=float)), errors="coerce").max()),
                "option_mtm_min": _float_or_blank(pd.to_numeric(eq_window.get("option_value", pd.Series(dtype=float)), errors="coerce").min()),
            }
        )
    return pd.DataFrame(rows)


def _annual_hedge_cost_rows(trades: pd.DataFrame, equity: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    t = trades.copy()
    t["date"] = pd.to_datetime(t["date"], errors="coerce")
    opens = t[(t.get("strategy") == "put_spread") & t["reason"].astype(str).str.contains("open", case=False, na=False)]
    if opens.empty:
        return pd.DataFrame()
    annual_cost = -opens.groupby(opens["date"].dt.year)["cash_flow"].sum()
    position_pnl = t.groupby("position_id")["cash_flow"].sum()
    position_year = opens.groupby("position_id")["date"].min().dt.year
    annual_recovered = position_pnl[position_pnl > 0].groupby(position_year).sum()
    annual_lost = (-position_pnl[position_pnl < 0]).groupby(position_year).sum()
    eq = _prepare_equity(equity)
    total_by_year = eq.groupby(eq["date"].dt.year)["total_equity"].first() if not eq.empty else pd.Series(dtype=float)
    rows: list[dict[str, Any]] = []
    for year, cost in annual_cost.items():
        base_equity = float(total_by_year.get(year, np.nan))
        recovered = float(annual_recovered.get(year, 0.0))
        lost = float(annual_lost.get(year, 0.0))
        rows.append(
            {
                "section": "annual_hedge_cost",
                "year": int(year),
                "annual_hedge_cost": float(cost),
                "hedge_cost_pct_of_portfolio_equity": _safe_ratio(float(cost), base_equity),
                "premium_lost": lost,
                "premium_recovered": recovered,
                "hedge_cost_but_no_crash_payoff": bool(float(cost) > 0 and recovered <= 0),
            }
        )
    rows.append(
        {
            "section": "hedge_cost_summary",
            "metric": "years_with_hedge_cost_but_no_crash_payoff",
            "value": int(sum(1 for row in rows if row.get("hedge_cost_but_no_crash_payoff"))),
        }
    )
    rows.append(
        {
            "section": "hedge_cost_summary",
            "metric": "premium_lost_total",
            "value": _float_or_blank(sum(float(row.get("premium_lost", 0.0)) for row in rows)),
        }
    )
    rows.append(
        {
            "section": "hedge_cost_summary",
            "metric": "premium_recovered_total",
            "value": _float_or_blank(sum(float(row.get("premium_recovered", 0.0)) for row in rows)),
        }
    )
    return pd.DataFrame(rows)


def _crash_effectiveness_rows(trades: pd.DataFrame, equity: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    t = trades.copy()
    if not t.empty:
        t["date"] = pd.to_datetime(t["date"], errors="coerce")
        t["expiry"] = pd.to_datetime(t["expiry"], errors="coerce")
    eq = _prepare_equity(equity)
    mk = _prepare_market(market)
    rows: list[dict[str, Any]] = []
    for name, (start_text, end_text) in CRASH_WINDOWS.items():
        start = pd.Timestamp(start_text)
        end = pd.Timestamp(end_text)
        eq_window = _window(eq, start, end)
        mk_window = _window(mk, start, end)
        active = _active_positions_in_window(t, start, end)
        option_change = _window_change(eq_window, "option_value")
        stock_change = _window_change(eq_window, "stock_equity")
        total_dd = _window_drawdown(eq_window)
        stock_dd = _series_drawdown(pd.to_numeric(eq_window.get("stock_equity", pd.Series(dtype=float)), errors="coerce"))
        rows.append(
            {
                "section": "crash_hedge_effectiveness",
                "period": name,
                "start": start_text,
                "end": end_text,
                "put_spread_exists": bool(not active.empty),
                "active_position_ids": ";".join(sorted(active["position_id"].astype(str).unique())) if not active.empty else "",
                "underlying_return": _underlying_return(mk_window),
                "option_mtm_change": _float_or_blank(option_change),
                "stock_equity_change": _float_or_blank(stock_change),
                "option_mtm_increased": bool(option_change > 0) if pd.notna(option_change) else "",
                "hedge_payoff_offset_stock_loss": _offset_ratio(option_change, stock_change),
                "portfolio_max_drawdown": _float_or_blank(total_dd),
                "stock_proxy_max_drawdown": _float_or_blank(stock_dd),
                "portfolio_drawdown_improved_vs_stock_proxy": bool(pd.notna(total_dd) and pd.notna(stock_dd) and total_dd > stock_dd),
            }
        )
    return pd.DataFrame(rows)


def _forced_exit_rows(trades: pd.DataFrame, lifecycle: pd.DataFrame, processed_data_dir: Path) -> pd.DataFrame:
    if lifecycle.empty:
        return pd.DataFrame()
    forced = lifecycle[lifecycle.get("issue", pd.Series(dtype=str)).astype(str) == "forced_unfilled_exit"].copy()
    if forced.empty:
        return pd.DataFrame()
    rows = []
    quote_probe = _load_option_quote_probe(processed_data_dir / "options.csv", forced)
    for row in forced.itertuples(index=False):
        pid = str(row.position_id)
        entry_rows = trades[trades.get("position_id", pd.Series(dtype=str)).astype(str) == pid] if not trades.empty else pd.DataFrame()
        remaining = int(getattr(row, "entry_legs", 0)) - int(getattr(row, "exit_legs", 0))
        rows.append(
            {
                "section": "forced_unfilled_exit",
                "position_id": pid,
                "entry_date": getattr(row, "entry_date", ""),
                "expiry": getattr(row, "expiry", ""),
                "exit_date": getattr(row, "exit_date", ""),
                "remaining_position": remaining,
                "why_no_normal_exit": "No generated exit trades before expiry; lifecycle audit forced an unfilled expiry marker.",
                "missing_valid_quote_on_exit_date": quote_probe.get(pid, "UNKNOWN"),
                "pnl_report_treatment": "Flag separately; do not silently treat as normal realized exit.",
                "recorded_cash_flow_sum": _float_or_blank(entry_rows["cash_flow"].sum()) if not entry_rows.empty else "",
            }
        )
    return pd.DataFrame(rows)


def _audit_context_rows(summary: pd.DataFrame, report_audit: pd.DataFrame, data_quality: pd.DataFrame, quote_quality: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if not summary.empty:
        for col in summary.columns:
            rows.append({"section": "summary_context", "metric": col, "value": summary[col].iloc[0]})
    for name, frame in [("report_audit", report_audit), ("data_quality", data_quality)]:
        if not frame.empty and "status" in frame:
            for status, count in frame["status"].value_counts().items():
                rows.append({"section": f"{name}_status_counts", "status": status, "count": int(count)})
    if not quote_quality.empty:
        overall = quote_quality[quote_quality.get("section") == "overall"] if "section" in quote_quality else pd.DataFrame()
        for item in overall.itertuples(index=False):
            rows.append(
                {
                    "section": "options_quote_quality_overall",
                    "quote_quality_status": getattr(item, "quote_quality_status", ""),
                    "count": int(getattr(item, "count", 0)),
                    "ratio": getattr(item, "ratio", ""),
                }
            )
    return pd.DataFrame(rows)


def _analysis_markdown(analysis: pd.DataFrame) -> str:
    data = _section_metric_map(analysis, "data_basis")
    trade_quality = _section_metric_map(analysis, "trade_quality")
    lifecycle = analysis[analysis["section"] == "forced_unfilled_exit"] if "section" in analysis else pd.DataFrame()
    positions = analysis[analysis["section"] == "position"] if "section" in analysis else pd.DataFrame()
    annual = analysis[analysis["section"] == "annual_hedge_cost"] if "section" in analysis else pd.DataFrame()
    crashes = analysis[analysis["section"] == "crash_hedge_effectiveness"] if "section" in analysis else pd.DataFrame()
    quote_dist = analysis[analysis["section"] == "trade_quote_quality_distribution"] if "section" in analysis else pd.DataFrame()

    lines = [
        "# Put Spread Real Data Diagnostic Analysis",
        "",
        "This report reads generated reports and processed data only. It does not change strategy rules, parameters, fills, or backtest results.",
        "",
        "## Data Basis",
        "",
        f"- Market date range: {data.get('market_date_range', '')}",
        f"- Options date range: {data.get('options_date_range', '')}",
        f"- Market rows: {data.get('market_rows', '')}",
        f"- Options rows: {data.get('options_rows', '')}",
        f"- Trades count: {data.get('trades_count', '')}",
        f"- Position count: {data.get('position_count', '')}",
        f"- forced_unfilled_exit count: {data.get('forced_unfilled_exit_count', '')}",
        "",
        "## Trade Quality",
        "",
    ]
    if quote_dist.empty:
        lines.append("- Quote quality distribution in trades: unavailable")
    else:
        dist = ", ".join(f"{row.quote_quality_status}={row.count}" for row in quote_dist.itertuples(index=False))
        lines.append(f"- Quote quality distribution in trades: {dist}")
    lines.extend(
        [
            f"- is_tradable_quote false count: {trade_quality.get('is_tradable_quote_false_count', '')}",
            f"- Average traded spread_pct: {trade_quality.get('average_spread_pct_traded_legs', '')}",
            f"- Max traded spread_pct: {trade_quality.get('max_spread_pct_traded_legs', '')}",
            f"- bid_ask_estimated count: {trade_quality.get('bid_ask_estimated_count', '')}",
            f"- iv_estimated count: {trade_quality.get('iv_estimated_count', '')}",
            f"- delta_estimated count: {trade_quality.get('delta_estimated_count', '')}",
            "",
            "## Position Behavior",
            "",
        ]
    )
    if positions.empty:
        lines.append("- No put-spread positions available.")
    else:
        for row in positions.itertuples(index=False):
            lines.append(
                f"- {row.position_id}: entry {row.entry_date}, expiry {row.expiry}, "
                f"long/short moneyness {row.long_put_moneyness}/{row.short_put_moneyness}, "
                f"realized_pnl {row.realized_pnl}, forced_unfilled_exit {row.forced_unfilled_exit}"
            )
    lines.extend(["", "## Hedge Cost", ""])
    if annual.empty:
        lines.append("- Annual hedge cost unavailable.")
    else:
        for row in annual.itertuples(index=False):
            lines.append(
                f"- {int(row.year)}: annual hedge cost {row.annual_hedge_cost}, "
                f"cost/equity {row.hedge_cost_pct_of_portfolio_equity}, "
                f"premium lost {row.premium_lost}, premium recovered {row.premium_recovered}"
            )
    lines.extend(["", "## Crash Windows", ""])
    if crashes.empty:
        lines.append("- Crash window diagnostics unavailable.")
    else:
        for row in crashes.itertuples(index=False):
            lines.append(
                f"- {row.period}: put_spread_exists={row.put_spread_exists}, "
                f"option_mtm_increased={row.option_mtm_increased}, "
                f"offset_ratio={row.hedge_payoff_offset_stock_loss}, "
                f"portfolio_max_drawdown={row.portfolio_max_drawdown}, "
                f"drawdown_improved_vs_stock_proxy={row.portfolio_drawdown_improved_vs_stock_proxy}"
            )
    lines.extend(["", "## forced_unfilled_exit", ""])
    if lifecycle.empty:
        lines.append("- None.")
    else:
        for row in lifecycle.itertuples(index=False):
            lines.append(
                f"- {row.position_id}: expiry {row.expiry}, exit marker {row.exit_date}, "
                f"remaining legs {row.remaining_position}, quote check {row.missing_valid_quote_on_exit_date}, "
                f"PnL treatment: {row.pnl_report_treatment}"
            )
    lines.extend(
        [
            "",
            "## Required Limitations",
            "",
            "- This is a diagnostic analysis, not an investment conclusion.",
            "- Results still depend on expiry calendar completeness, VIX proxy, and forced_unfilled_exit handling.",
            "- Do not optimize parameters based on this report.",
        ]
    )
    return "\n".join(lines) + "\n"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _read_market(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        cols = pd.read_csv(path, nrows=0).columns
        usecols = [col for col in ["date", "tx_close", "txf_close", "tx_open", "tx_high", "tx_low"] if col in cols]
        return pd.read_csv(path, usecols=usecols)
    except Exception:
        return pd.DataFrame()


def _read_option_dates(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, usecols=["date"])
    except Exception:
        return pd.DataFrame()


def _row_count_from_quality(data_quality: pd.DataFrame, file_name: str) -> int | str:
    if data_quality.empty:
        return ""
    rows = data_quality[(data_quality.get("file") == file_name) & (data_quality.get("check") == "has_rows")]
    if rows.empty:
        return ""
    detail = str(rows["detail"].iloc[0])
    if "rows=" not in detail:
        return detail
    try:
        return int(detail.split("rows=", 1)[1].split(",", 1)[0])
    except ValueError:
        return detail


def _date_range(frame: pd.DataFrame, column: str) -> str:
    if frame.empty or column not in frame:
        return ""
    dates = pd.to_datetime(frame[column], errors="coerce").dropna()
    if dates.empty:
        return ""
    return f"{dates.min().date()} to {dates.max().date()}"


def _position_count(trades: pd.DataFrame, lifecycle: pd.DataFrame) -> int:
    ids = set()
    if not trades.empty and "position_id" in trades:
        ids |= set(trades["position_id"].dropna().astype(str))
    if not lifecycle.empty and "position_id" in lifecycle:
        ids |= set(lifecycle["position_id"].dropna().astype(str))
    return len(ids)


def _prepare_equity(equity: pd.DataFrame) -> pd.DataFrame:
    if equity.empty or "date" not in equity:
        return pd.DataFrame()
    out = equity.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out.dropna(subset=["date"]).sort_values("date")


def _prepare_market(market: pd.DataFrame) -> pd.DataFrame:
    if market.empty or "date" not in market:
        return pd.DataFrame()
    out = market.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out.dropna(subset=["date"]).sort_values("date")


def _window(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if frame.empty or "date" not in frame or pd.isna(start) or pd.isna(end):
        return pd.DataFrame()
    return frame[(frame["date"] >= start) & (frame["date"] <= end)].copy()


def _window_drawdown(frame: pd.DataFrame) -> float | str:
    if frame.empty or "total_equity" not in frame:
        return ""
    return _float_or_blank(_series_drawdown(pd.to_numeric(frame["total_equity"], errors="coerce")))


def _series_drawdown(series: pd.Series) -> float:
    values = series.dropna()
    if values.empty:
        return np.nan
    return float((values / values.cummax() - 1.0).min())


def _underlying_return(market_window: pd.DataFrame) -> float | str:
    if market_window.empty:
        return ""
    for col in ["txf_close", "tx_close"]:
        if col in market_window:
            values = pd.to_numeric(market_window[col], errors="coerce").dropna()
            if len(values) >= 2 and values.iloc[0] != 0:
                return float(values.iloc[-1] / values.iloc[0] - 1.0)
    return ""


def _window_change(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return np.nan
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if len(values) < 2:
        return np.nan
    return float(values.iloc[-1] - values.iloc[0])


def _active_positions_in_window(trades: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    active_ids = []
    for pid, group in trades.groupby("position_id"):
        opens = group[group["reason"].astype(str).str.contains("open", case=False, na=False)]
        if opens.empty:
            continue
        entry = opens["date"].min()
        expiry = group["expiry"].dropna().min()
        exits = group[~group.index.isin(opens.index)]
        exit_date = exits["date"].max() if not exits.empty else expiry
        if pd.notna(entry) and pd.notna(exit_date) and entry <= end and exit_date >= start:
            active_ids.append(pid)
    return trades[trades["position_id"].isin(active_ids)].copy()


def _offset_ratio(option_change: float, stock_change: float) -> float | str:
    if pd.isna(option_change) or pd.isna(stock_change) or stock_change >= 0:
        return ""
    return float(option_change / abs(stock_change))


def _exit_reason(exits: pd.DataFrame, event: dict[str, Any]) -> str:
    if not exits.empty and "reason" in exits:
        return ";".join(sorted(exits["reason"].dropna().astype(str).unique()))
    if event.get("issue"):
        return str(event.get("issue"))
    return ""


def _lifecycle_map(lifecycle: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if lifecycle.empty or "position_id" not in lifecycle:
        return {}
    return {str(row["position_id"]): row.to_dict() for _, row in lifecycle.iterrows()}


def _first_numeric(frame: pd.DataFrame, columns: list[str]) -> float:
    for col in columns:
        if col in frame:
            values = pd.to_numeric(frame[col], errors="coerce").dropna()
            if not values.empty:
                return float(values.iloc[0])
    return np.nan


def _safe_ratio(numerator: float, denominator: float) -> float | str:
    if pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
        return ""
    return float(numerator / denominator)


def _float_or_blank(value: Any) -> float | str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(value):
        return ""
    return value


def _date_text(value: Any) -> str:
    ts = pd.to_datetime(value, errors="coerce")
    return "" if pd.isna(ts) else str(ts.date())


def _bool_series(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.Series(dtype=bool)
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin({"true", "1", "yes", "y"})


def _section_metric_map(analysis: pd.DataFrame, section: str) -> dict[str, Any]:
    if analysis.empty or "section" not in analysis:
        return {}
    frame = analysis[(analysis["section"] == section) & analysis.get("metric", pd.Series(dtype=str)).notna()]
    if frame.empty:
        return {}
    return dict(zip(frame["metric"], frame.get("value", pd.Series(dtype=object))))


def _load_option_quote_probe(path: Path, forced: pd.DataFrame) -> dict[str, str]:
    if forced.empty or not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        needed_dates = set(pd.to_datetime(forced["exit_date"], errors="coerce").dropna().dt.strftime("%Y-%m-%d"))
        if not needed_dates:
            return {}
        cols = pd.read_csv(path, nrows=0).columns
        usecols = [col for col in ["date", "expiry", "cp", "strike", "quote_quality_status", "is_tradable_quote"] if col in cols]
        if {"date", "expiry", "cp", "strike"} - set(usecols):
            return {}
        options = pd.read_csv(path, usecols=usecols)
        options = options[options["date"].astype(str).isin(needed_dates)].copy()
    except Exception:
        return {}
    result: dict[str, str] = {}
    for row in forced.itertuples(index=False):
        pid = str(row.position_id)
        exit_date = str(getattr(row, "exit_date", ""))
        expiry = str(getattr(row, "expiry", ""))
        matches = options[(options["date"].astype(str) == exit_date) & (options["expiry"].astype(str) == expiry)]
        if matches.empty:
            result[pid] = "NO_QUOTE_FOR_EXIT_DATE_EXPIRY"
        elif "quote_quality_status" in matches and "is_tradable_quote" in matches:
            valid = matches[(matches["quote_quality_status"].astype(str) == "VALID") & _bool_series(matches["is_tradable_quote"])]
            result[pid] = "HAS_VALID_QUOTE" if not valid.empty else "NO_VALID_QUOTE"
        else:
            result[pid] = "QUOTE_QUALITY_COLUMNS_UNAVAILABLE"
    return result
