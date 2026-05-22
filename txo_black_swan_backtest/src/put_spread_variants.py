"""Diagnostic put-spread insurance variants.

These variants compare coarse, pre-defined insurance schedules. They do not
search parameters, select a best result, or change execution semantics.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .contracts import Leg, Position, StrategyState
from .data_loader import load_macro_factors, load_market, load_options, load_portfolio
from .execution import fill_price, is_stress_day, trade_contract
from .indicators import add_market_indicators
from .macro_regime import add_macro_regime_indicators, macro_restrictions
from .metrics import max_drawdown
from .portfolio import make_state
from .strategies import BlackSwanStateMachine
from .utils import year_key


VARIANTS = ["current_signal_based", "quarterly_base_insurance", "base_plus_signal_boost"]
CRASH_WINDOWS: dict[str, tuple[str, str]] = {
    "2008": ("2008-01-01", "2008-12-31"),
    "2011": ("2011-01-01", "2011-12-31"),
    "2015": ("2015-06-01", "2015-12-31"),
    "2018": ("2018-01-01", "2018-12-31"),
    "2020": ("2020-01-01", "2020-12-31"),
    "2022": ("2022-01-01", "2022-12-31"),
}


class PutSpreadVariantStateMachine(BlackSwanStateMachine):
    """State machine with diagnostic-only Put Spread entry variants."""

    def __init__(self, *args, variant_name: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.variant_name = variant_name
        self.checked_quarters: set[str] = set()

    def _check_put_entry(self, row, stock_equity: float) -> None:
        if self.variant_name == "current_signal_based":
            super()._check_put_entry(row, stock_equity)
            return
        if self.variant_name == "quarterly_base_insurance":
            self._check_quarterly_base(row, stock_equity, allow_signal_boost=False)
            return
        if self.variant_name == "base_plus_signal_boost":
            self._check_quarterly_base(row, stock_equity, allow_signal_boost=True)
            return
        raise ValueError(f"Unknown put-spread variant: {self.variant_name}")

    def _check_quarterly_base(self, row, stock_equity: float, allow_signal_boost: bool) -> None:
        date = pd.Timestamp(row.date)
        quarter_key = f"{date.year}Q{date.quarter}"
        if quarter_key in self.checked_quarters:
            return
        self.checked_quarters.add(quarter_key)
        if self._open_positions("put_spread"):
            return
        year = year_key(row.date)
        macro_state = str(getattr(row, "macro_state", "NORMAL"))
        restriction = macro_restrictions(macro_state)
        budget_cap = float(self.config["max_annual_hedge_budget_pct"]) * float(restriction["hedge_budget_multiplier"]) * stock_equity
        used = self.annual_hedge_spend.get(year, 0.0)
        remaining = max(0.0, budget_cap - used)
        if remaining <= 0:
            return
        base_budget = float(self.config.get("base_annual_hedge_budget_pct", self.config["max_annual_hedge_budget_pct"])) * stock_equity
        entry_budget = min(remaining, base_budget / 4.0)
        signal_ok = self._original_signal_ok(row)
        if allow_signal_boost and signal_ok:
            entry_budget = remaining
        self._open_put_spread_with_budget(row, stock_equity, entry_budget, "quarterly_base_put_spread" if not signal_ok else "quarterly_signal_boost_put_spread")

    def _original_signal_ok(self, row) -> bool:
        if any(pd.isna(x) for x in [row.ma200, row.ret_126d, row.vix_percentile_3y]):
            return False
        macro_state = str(getattr(row, "macro_state", "NORMAL"))
        vix_threshold = 40.0 if macro_state == "STAGFLATION_PRESSURE" and float(getattr(row, "ValuationRiskIndex", 0.0)) > 70.0 else 30.0
        return bool(
            row.tx_close > row.ma200
            and row.ret_126d > 0.20
            and row.vix_percentile_3y < vix_threshold
            and int(row.event_flag) == 0
        )

    def _open_put_spread_with_budget(self, row, stock_equity: float, entry_budget: float, reason: str) -> None:
        if entry_budget <= 0:
            return
        macro_state = str(getattr(row, "macro_state", "NORMAL"))
        restriction = macro_restrictions(macro_state)
        low_vix = pd.notna(row.vix_percentile_3y) and row.vix_percentile_3y < 20.0
        long_m = self.put_params["long_put_moneyness_low_vix"] if low_vix else self.put_params["long_put_moneyness"]
        short_m = self.put_params["short_put_moneyness_low_vix"] if low_vix else self.put_params["short_put_moneyness"]
        dte_min = int(self.put_params["target_dte_min"])
        dte_max = int(self.put_params["target_dte_max"]) + int(restriction["extend_put_dte"])
        date = pd.Timestamp(row.date)
        long_put = self.selector.nearest_strike(date, "P", row.txf_close * long_m, dte_min, dte_max)
        if long_put is None:
            return
        short_put = self.selector.nearest_strike(
            date,
            "P",
            row.txf_close * short_m,
            dte_min,
            dte_max,
            expiry=pd.Timestamp(long_put.expiry),
        )
        if short_put is None or short_put.strike >= long_put.strike:
            return
        stress = is_stress_day(row, self.config)
        long_px = fill_price(long_put, "BUY", self.config, stress)
        short_px = fill_price(short_put, "SELL", self.config, stress)
        per_spread_debit = (long_px - short_px) * float(self.config["txo_point_value"])
        per_spread_cost = per_spread_debit + 2 * float(self.config["commission_per_contract_per_side"])
        if per_spread_cost <= 0:
            return
        notional_contracts = max(
            1,
            int((stock_equity * float(self.config["portfolio_beta"])) / (row.txf_close * float(self.config["txo_point_value"])) * 0.5),
        )
        qty = min(notional_contracts, int(entry_budget // per_spread_cost))
        if qty <= 0:
            return
        pos_id = f"PS-{next(self.position_counter)}"
        fill1, tr1 = trade_contract(str(date.date()), pos_id, "put_spread", long_put, "BUY", qty, self.config, reason, stress)
        fill2, tr2 = trade_contract(str(date.date()), pos_id, "put_spread", short_put, "SELL", qty, self.config, reason, stress)
        self.cash += fill1.cash_flow + fill2.cash_flow
        self.trades.extend([tr1, tr2])
        spend = -(fill1.cash_flow + fill2.cash_flow)
        year = year_key(row.date)
        self.annual_hedge_spend[year] = self.annual_hedge_spend.get(year, 0.0) + max(0.0, spend)
        self.positions.append(
            Position(
                id=pos_id,
                strategy="put_spread",
                entry_date=str(date.date()),
                legs=[Leg(long_put, qty, fill1.price), Leg(short_put, -qty, fill2.price)],
                quantity=qty,
                entry_underlying=float(row.tx_close),
                entry_cost=spend,
                max_loss=spend,
                notes={"remaining_entry_cost": spend, "variant": self.variant_name},
            )
        )
        self.state = StrategyState.HEDGE_ON


def run_put_spread_variants(data_dir: Path, report_dir: Path, config: dict, put_params: dict, ic_params: dict) -> pd.DataFrame:
    """Run all fixed variants and write comparison reports."""

    report_dir.mkdir(parents=True, exist_ok=True)
    market = load_market(data_dir)
    if market.empty:
        out = pd.DataFrame([{"section": "error", "status": "FAIL", "detail": "market.csv unavailable"}])
        out.to_csv(report_dir / "put_spread_variant_comparison.csv", index=False)
        (report_dir / "put_spread_variant_comparison.md").write_text(_markdown(out), encoding="utf-8")
        return out
    market = add_market_indicators(market)
    market = add_macro_regime_indicators(market, load_macro_factors(data_dir))
    load_config = config.copy()
    load_config["runtime_mode"] = "put_spread_only"
    options = load_options(data_dir, market, load_config)
    portfolio = load_portfolio(data_dir, market, config)

    rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        engine = PutSpreadVariantStateMachine(market, options, portfolio, config, put_params, ic_params, mode="put_spread_only", variant_name=variant)
        equity, trades = engine.run()
        lifecycle = pd.DataFrame(equity.attrs.get("position_lifecycle_events", []))
        rows.extend(_variant_summary_rows(variant, equity, trades, lifecycle, config))
        rows.extend(_annual_budget_rows(variant, equity, trades, config))
        rows.extend(_crash_rows(variant, equity, trades, lifecycle, market))
        rows.extend(_quote_audit_rows(variant, trades))
        rows.extend(_expiry_audit_rows(variant, trades))
    comparison = pd.DataFrame(rows)
    comparison.to_csv(report_dir / "put_spread_variant_comparison.csv", index=False)
    (report_dir / "put_spread_variant_comparison.md").write_text(_markdown(comparison), encoding="utf-8")
    return comparison


def _variant_summary_rows(variant: str, equity: pd.DataFrame, trades: pd.DataFrame, lifecycle: pd.DataFrame, config: dict) -> list[dict[str, Any]]:
    position_count = int(trades["position_id"].nunique()) if not trades.empty else 0
    forced_count = int((lifecycle.get("issue", pd.Series(dtype=str)).astype(str) == "forced_unfilled_exit").sum()) if not lifecycle.empty else 0
    realized = float(trades["cash_flow"].sum()) if not trades.empty else 0.0
    cost_drag = float(trades["cost"].sum()) if not trades.empty and "cost" in trades else 0.0
    return [
        {
            "section": "variant_summary",
            "variant": variant,
            "position_count": position_count,
            "forced_unfilled_exit_count": forced_count,
            "realized_hedge_payoff": realized,
            "total_equity_max_drawdown": max_drawdown(equity["total_equity"]) if not equity.empty else "",
            "cost_drag_total": cost_drag,
        },
        {
            "section": "variant_audit",
            "variant": variant,
            "check": "forced_unfilled_exit_count",
            "status": "WARN" if forced_count else "PASS",
            "detail": f"forced_unfilled_exit_count={forced_count}",
        },
    ]


def _annual_budget_rows(variant: str, equity: pd.DataFrame, trades: pd.DataFrame, config: dict) -> list[dict[str, Any]]:
    if trades.empty:
        return []
    t = trades.copy()
    t["date"] = pd.to_datetime(t["date"], errors="coerce")
    opens = t[(t["strategy"] == "put_spread") & t["reason"].astype(str).str.contains("open|quarterly", case=False, regex=True, na=False)]
    if opens.empty:
        return []
    eq = equity.copy()
    eq["date"] = pd.to_datetime(eq["date"], errors="coerce")
    yearly_equity = eq.groupby(eq["date"].dt.year)["total_equity"].first()
    annual_cost = -opens.groupby(opens["date"].dt.year)["cash_flow"].sum()
    rows: list[dict[str, Any]] = []
    for year, cost in annual_cost.items():
        base_equity = float(yearly_equity.get(year, np.nan))
        budget_cap = float(config["max_annual_hedge_budget_pct"]) * base_equity if pd.notna(base_equity) else np.nan
        breach = bool(pd.notna(budget_cap) and float(cost) > budget_cap + 1e-8)
        rows.append(
            {
                "section": "annual_budget_usage",
                "variant": variant,
                "year": int(year),
                "annual_hedge_cost": float(cost),
                "annual_hedge_cost_pct_equity": _safe_ratio(float(cost), base_equity),
                "cost_drag_in_no_crash_year": 0.0 if int(year) in {2008, 2011, 2015, 2018, 2020, 2022} else float(cost),
                "annual_budget_cap": budget_cap,
                "budget_usage_ratio": _safe_ratio(float(cost), budget_cap),
                "annual_budget_breach": breach,
            }
        )
    breach_count = int(sum(bool(row["annual_budget_breach"]) for row in rows))
    rows.append(
        {
            "section": "variant_audit",
            "variant": variant,
            "check": "annual_budget_breach",
            "status": "FAIL" if breach_count else "PASS",
            "detail": f"years={breach_count}",
        }
    )
    return rows


def _crash_rows(variant: str, equity: pd.DataFrame, trades: pd.DataFrame, lifecycle: pd.DataFrame, market: pd.DataFrame) -> list[dict[str, Any]]:
    positions = _position_windows(trades, lifecycle)
    eq = equity.copy()
    eq["date"] = pd.to_datetime(eq["date"], errors="coerce")
    mk = market.copy()
    mk["date"] = pd.to_datetime(mk["date"], errors="coerce")
    rows: list[dict[str, Any]] = []
    for period, (start_text, end_text) in CRASH_WINDOWS.items():
        start = pd.Timestamp(start_text)
        end = pd.Timestamp(end_text)
        eq_window = eq[(eq["date"] >= start) & (eq["date"] <= end)]
        market_window = mk[(mk["date"] >= start) & (mk["date"] <= end)]
        active = [pos for pos in positions if pos["entry_date"] <= end and pos["exit_date"] >= start]
        total_days = int(market_window["date"].nunique()) if not market_window.empty else 0
        covered_days = _covered_days(active, start, end, market_window["date"] if not market_window.empty else pd.Series(dtype="datetime64[ns]"))
        rows.append(
            {
                "section": "crash_window",
                "variant": variant,
                "period": period,
                "crash_window_coverage_days": covered_days,
                "crash_window_coverage_ratio": _safe_ratio(covered_days, total_days),
                "active_position_ids": ";".join(pos["position_id"] for pos in active),
                "option_mtm_during_crash": _window_change(eq_window, "option_value"),
                "market_max_drawdown": _max_drawdown(market_window, "tx_close"),
            }
        )
    return rows


def _quote_audit_rows(variant: str, trades: pd.DataFrame) -> list[dict[str, Any]]:
    if trades.empty:
        return [
            {"section": "quote_quality_distribution", "variant": variant, "quote_quality_status": "NO_TRADES", "count": 0},
            {"section": "variant_audit", "variant": variant, "check": "non_valid_quote_trades", "status": "PASS", "detail": "legs=0"},
        ]
    status = trades.get("quote_quality_status", pd.Series(["UNKNOWN"] * len(trades))).fillna("UNKNOWN").astype(str)
    rows = [{"section": "quote_quality_distribution", "variant": variant, "quote_quality_status": item, "count": int(count)} for item, count in status.value_counts().items()]
    non_valid = int((status != "VALID").sum())
    rows.append(
        {
            "section": "variant_audit",
            "variant": variant,
            "check": "non_valid_quote_trades",
            "status": "FAIL" if non_valid else "PASS",
            "detail": f"legs={non_valid}",
        }
    )
    return rows


def _expiry_audit_rows(variant: str, trades: pd.DataFrame) -> list[dict[str, Any]]:
    if trades.empty:
        return [{"section": "variant_audit", "variant": variant, "check": "trade_date_after_expiry", "status": "PASS", "detail": "rows=0"}]
    trade_date = pd.to_datetime(trades["date"], errors="coerce")
    expiry = pd.to_datetime(trades["expiry"], errors="coerce")
    after = int((trade_date > expiry).sum())
    return [
        {
            "section": "variant_audit",
            "variant": variant,
            "check": "trade_date_after_expiry",
            "status": "FAIL" if after else "PASS",
            "detail": f"rows={after}",
        }
    ]


def _markdown(comparison: pd.DataFrame) -> str:
    summary = comparison[comparison["section"] == "variant_summary"] if not comparison.empty else pd.DataFrame()
    crash = comparison[comparison["section"] == "crash_window"] if not comparison.empty else pd.DataFrame()
    budget = comparison[comparison["section"] == "annual_budget_usage"] if not comparison.empty else pd.DataFrame()
    audit = comparison[comparison["section"] == "variant_audit"] if not comparison.empty else pd.DataFrame()
    lines = [
        "# Put Spread Variant Comparison",
        "",
        "This report compares three pre-defined insurance regimes. It does not rank parameter sets and does not optimize thresholds.",
        "",
        "## Variant Summary",
        "",
    ]
    if summary.empty:
        lines.append("- No variant summary generated.")
    else:
        for row in summary.itertuples(index=False):
            lines.append(
                f"- {row.variant}: positions={row.position_count}, forced_unfilled_exit={row.forced_unfilled_exit_count}, "
                f"realized_hedge_payoff={row.realized_hedge_payoff}, max_drawdown={row.total_equity_max_drawdown}"
            )
    lines.extend(["", "## Crash Coverage", ""])
    if crash.empty:
        lines.append("- No crash coverage generated.")
    else:
        for variant in VARIANTS:
            sub = crash[crash["variant"] == variant]
            ratios = ", ".join(f"{row.period}={row.crash_window_coverage_ratio}" for row in sub.itertuples(index=False))
            lines.append(f"- {variant}: {ratios}")
    lines.extend(["", "## Annual Hedge Cost", ""])
    if budget.empty:
        lines.append("- No annual hedge costs generated.")
    else:
        for variant in VARIANTS:
            sub = budget[budget["variant"] == variant]
            costs = ", ".join(f"{int(row.year)}={row.annual_hedge_cost}" for row in sub.itertuples(index=False) if pd.notna(row.year))
            lines.append(f"- {variant}: {costs}")
    lines.extend(["", "## Audit", ""])
    if audit.empty:
        lines.append("- No audit rows generated.")
    else:
        for row in audit.itertuples(index=False):
            lines.append(f"- {row.variant}.{row.check}: {row.status} ({row.detail})")
    lines.extend(
        [
            "",
            "## Required Limitations",
            "",
            "- This is a diagnostic comparison of fixed insurance regimes, not an investment conclusion.",
            "- It does not recommend parameter changes.",
            "- Do not optimize entry thresholds based on crash windows.",
            "- The comparison table intentionally does not rank variants.",
        ]
    )
    return "\n".join(lines) + "\n"


def _position_windows(trades: pd.DataFrame, lifecycle: pd.DataFrame) -> list[dict[str, Any]]:
    if trades.empty:
        return []
    t = trades.copy()
    t["date"] = pd.to_datetime(t["date"], errors="coerce")
    t["expiry"] = pd.to_datetime(t["expiry"], errors="coerce")
    events = {str(row["position_id"]): row.to_dict() for _, row in lifecycle.iterrows()} if not lifecycle.empty and "position_id" in lifecycle else {}
    windows = []
    for pid, group in t.groupby("position_id"):
        opens = group[group["reason"].astype(str).str.contains("open|quarterly", case=False, regex=True, na=False)]
        exits = group[~group.index.isin(opens.index)]
        if opens.empty:
            continue
        entry = opens["date"].min()
        expiry = group["expiry"].dropna().min()
        event_exit = pd.to_datetime(events.get(str(pid), {}).get("exit_date", pd.NaT), errors="coerce")
        exit_date = event_exit if pd.notna(event_exit) else (exits["date"].max() if not exits.empty else expiry)
        windows.append({"position_id": str(pid), "entry_date": entry, "expiry": expiry, "exit_date": exit_date})
    return windows


def _covered_days(positions: list[dict[str, Any]], start: pd.Timestamp, end: pd.Timestamp, dates: pd.Series) -> int:
    if not positions or dates.empty:
        return 0
    covered = pd.Series(False, index=dates.index)
    for pos in positions:
        covered |= (dates >= max(start, pos["entry_date"])) & (dates <= min(end, pos["exit_date"]))
    return int(covered.sum())


def _window_change(frame: pd.DataFrame, column: str) -> float | str:
    if frame.empty or column not in frame:
        return ""
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if len(values) < 2:
        return ""
    return float(values.iloc[-1] - values.iloc[0])


def _max_drawdown(frame: pd.DataFrame, column: str) -> float | str:
    if frame.empty or column not in frame:
        return ""
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return ""
    return float((values / values.cummax() - 1.0).min())


def _safe_ratio(numerator: float, denominator: float) -> float | str:
    if pd.isna(denominator) or denominator == 0:
        return ""
    return float(numerator / denominator)
