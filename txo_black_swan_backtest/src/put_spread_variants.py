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
    variant_runs: list[dict[str, Any]] = []
    for variant in VARIANTS:
        engine = PutSpreadVariantStateMachine(market, options, portfolio, config, put_params, ic_params, mode="put_spread_only", variant_name=variant)
        equity, trades = engine.run()
        lifecycle = pd.DataFrame(equity.attrs.get("position_lifecycle_events", []))
        variant_runs.append({"variant": variant, "equity": equity, "trades": trades, "lifecycle": lifecycle})
        rows.extend(_variant_summary_rows(variant, equity, trades, lifecycle, config))
        rows.extend(_annual_budget_rows(variant, equity, trades, config))
        rows.extend(_crash_rows(variant, equity, trades, lifecycle, market))
        rows.extend(_quote_audit_rows(variant, trades))
        rows.extend(_expiry_audit_rows(variant, trades))
    comparison = pd.DataFrame(rows)
    comparison.to_csv(report_dir / "put_spread_variant_comparison.csv", index=False)
    (report_dir / "put_spread_variant_comparison.md").write_text(_markdown(comparison), encoding="utf-8")
    forced = _forced_exit_liquidity_analysis(variant_runs, options, config, put_params)
    forced.to_csv(report_dir / "forced_exit_liquidity_analysis.csv", index=False)
    (report_dir / "forced_exit_liquidity_analysis.md").write_text(_forced_markdown(forced), encoding="utf-8")
    feasibility = _entry_exit_feasibility_analysis(variant_runs, options, config, put_params)
    feasibility.to_csv(report_dir / "entry_exit_feasibility_analysis.csv", index=False)
    (report_dir / "entry_exit_feasibility_analysis.md").write_text(_feasibility_markdown(feasibility), encoding="utf-8")
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


def _forced_exit_liquidity_analysis(variant_runs: list[dict[str, Any]], options: pd.DataFrame, config: dict, put_params: dict) -> pd.DataFrame:
    opt = options.copy()
    if not opt.empty:
        opt["date"] = pd.to_datetime(opt["date"], errors="coerce")
        opt["expiry"] = pd.to_datetime(opt["expiry"], errors="coerce")
    rows: list[dict[str, Any]] = []
    all_positions: list[dict[str, Any]] = []
    for run in variant_runs:
        variant = str(run["variant"])
        trades = run["trades"].copy()
        lifecycle = run["lifecycle"].copy()
        positions = _position_metadata(variant, trades, lifecycle)
        all_positions.extend(positions)
        forced_positions = [pos for pos in positions if pos["forced_unfilled_exit"]]
        for pos in forced_positions:
            window = _exit_window_availability_rows(pos, opt, config)
            rows.extend(window)
            reason = _classify_forced_exit(pos, window, config, put_params)
            rows.append({**pos, "section": "forced_position", "failure_reason": reason})
    rows.extend(_forced_aggregate_rows(all_positions, rows))
    rows.extend(_normal_vs_forced_comparison_rows(all_positions, opt, config))
    return pd.DataFrame(rows)


def _position_metadata(variant: str, trades: pd.DataFrame, lifecycle: pd.DataFrame) -> list[dict[str, Any]]:
    if trades.empty:
        return []
    t = trades.copy()
    t["date"] = pd.to_datetime(t["date"], errors="coerce")
    t["expiry"] = pd.to_datetime(t["expiry"], errors="coerce")
    lifecycle_map = {str(row["position_id"]): row.to_dict() for _, row in lifecycle.iterrows()} if not lifecycle.empty and "position_id" in lifecycle else {}
    positions: list[dict[str, Any]] = []
    for pid, group in t.groupby("position_id"):
        opens = group[group["reason"].astype(str).str.contains("open|quarterly", case=False, regex=True, na=False)]
        exits = group[~group.index.isin(opens.index)]
        if opens.empty:
            continue
        long = opens[(opens["action"] == "BUY") & (opens["cp"] == "P")].sort_values("strike", ascending=False)
        short = opens[(opens["action"] == "SELL") & (opens["cp"] == "P")].sort_values("strike")
        if long.empty or short.empty:
            continue
        long_row = long.iloc[0]
        short_row = short.iloc[0]
        event = lifecycle_map.get(str(pid), {})
        forced = str(event.get("issue", "")) == "forced_unfilled_exit"
        entry_date = pd.to_datetime(opens["date"].min())
        expiry = pd.to_datetime(opens["expiry"].min())
        event_exit = pd.to_datetime(event.get("exit_date", pd.NaT), errors="coerce")
        exit_date = event_exit if pd.notna(event_exit) else (exits["date"].max() if not exits.empty else expiry)
        entry_underlying = _first_numeric(opens, ["txf_close_at_trade", "underlying_price_at_trade"])
        positions.append(
            {
                "variant": variant,
                "position_id": str(pid),
                "entry_date": str(entry_date.date()),
                "expiry": str(expiry.date()),
                "exit_date": "" if pd.isna(exit_date) else str(pd.Timestamp(exit_date).date()),
                "entry_dte": _float_or_blank(long_row.get("dte_at_trade")),
                "long_put_strike": float(long_row["strike"]),
                "short_put_strike": float(short_row["strike"]),
                "long_put_moneyness": _safe_ratio(float(long_row["strike"]), entry_underlying),
                "short_put_moneyness": _safe_ratio(float(short_row["strike"]), entry_underlying),
                "quantity": int(abs(long_row["quantity"])),
                "net_premium_paid": float(-opens["cash_flow"].sum()),
                "forced_unfilled_exit": forced,
                "normal_exit": bool(not forced and not exits.empty),
            }
        )
    return positions


def _exit_window_availability_rows(pos: dict[str, Any], options: pd.DataFrame, config: dict) -> list[dict[str, Any]]:
    expiry = pd.Timestamp(pos["expiry"])
    start = expiry - pd.Timedelta(days=30)
    dates = pd.date_range(start, expiry, freq="D")
    relevant = options[
        (options["expiry"] == expiry)
        & (options["cp"] == "P")
        & (options["date"] >= start)
        & (options["date"] <= expiry)
        & (np.isclose(options["strike"], float(pos["long_put_strike"])) | np.isclose(options["strike"], float(pos["short_put_strike"])))
    ].copy()
    rows: list[dict[str, Any]] = []
    for date in dates:
        long_quote = _leg_quote(relevant, date, pos["long_put_strike"])
        short_quote = _leg_quote(relevant, date, pos["short_put_strike"])
        rows.append(
            {
                "section": "exit_window_day",
                "variant": pos["variant"],
                "position_id": pos["position_id"],
                "date": str(date.date()),
                "entry_date": pos["entry_date"],
                "expiry": pos["expiry"],
                "dte": int((expiry - date).days),
                "long_leg_quote_exists": long_quote is not None,
                "short_leg_quote_exists": short_quote is not None,
                "long_leg_quote_quality_status": _quote_value(long_quote, "quote_quality_status"),
                "short_leg_quote_quality_status": _quote_value(short_quote, "quote_quality_status"),
                "long_leg_is_tradable_quote": _quote_bool(long_quote, "is_tradable_quote"),
                "short_leg_is_tradable_quote": _quote_bool(short_quote, "is_tradable_quote"),
                "long_leg_volume": _quote_float(long_quote, "volume"),
                "short_leg_volume": _quote_float(short_quote, "volume"),
                "long_leg_open_interest": _quote_float(long_quote, "open_interest"),
                "short_leg_open_interest": _quote_float(short_quote, "open_interest"),
                "long_leg_spread_pct": _quote_float(long_quote, "spread_pct"),
                "short_leg_spread_pct": _quote_float(short_quote, "spread_pct"),
                "both_legs_liquid": _both_liquid(long_quote, short_quote, config),
            }
        )
    return rows


def _classify_forced_exit(pos: dict[str, Any], window_rows: list[dict[str, Any]], config: dict, put_params: dict) -> str:
    if pd.Timestamp(pos["expiry"]) < pd.Timestamp(pos["entry_date"]) or float(pos.get("entry_dte", -1)) < 0:
        return "BAD_EXPIRY"
    window = pd.DataFrame(window_rows)
    if window.empty:
        return "UNKNOWN"
    if not window["long_leg_quote_exists"].any():
        return "MISSING_LONG_LEG_QUOTE"
    if not window["short_leg_quote_exists"].any():
        return "MISSING_SHORT_LEG_QUOTE"
    dte_window = window[pd.to_numeric(window["dte"], errors="coerce") < int(put_params.get("exit_dte", 14))]
    if not dte_window.empty and dte_window["both_legs_liquid"].any():
        return "DTE_EXIT_WINDOW_MISSED"
    if _has_non_valid(window, "long"):
        return "NON_VALID_LONG_QUOTE"
    if _has_non_valid(window, "short"):
        return "NON_VALID_SHORT_QUOTE"
    min_volume = float(config.get("wide_spread_volume_threshold", 50))
    min_oi = float(config.get("min_open_interest", 100))
    if pd.to_numeric(window["long_leg_volume"], errors="coerce").fillna(0).lt(min_volume).any():
        return "LOW_LONG_VOLUME"
    if pd.to_numeric(window["short_leg_volume"], errors="coerce").fillna(0).lt(min_volume).any():
        return "LOW_SHORT_VOLUME"
    if pd.to_numeric(window["long_leg_open_interest"], errors="coerce").fillna(0).lt(min_oi).any():
        return "LOW_LONG_OPEN_INTEREST"
    if pd.to_numeric(window["short_leg_open_interest"], errors="coerce").fillna(0).lt(min_oi).any():
        return "LOW_SHORT_OPEN_INTEREST"
    if pd.to_numeric(window["long_leg_spread_pct"], errors="coerce").gt(0.5).any():
        return "EXTREME_LONG_SPREAD"
    if pd.to_numeric(window["short_leg_spread_pct"], errors="coerce").gt(0.5).any():
        return "EXTREME_SHORT_SPREAD"
    return "UNKNOWN"


def _forced_aggregate_rows(all_positions: list[dict[str, Any]], analysis_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    positions = pd.DataFrame(all_positions)
    forced = pd.DataFrame([row for row in analysis_rows if row.get("section") == "forced_position"])
    window = pd.DataFrame([row for row in analysis_rows if row.get("section") == "exit_window_day"])
    rows: list[dict[str, Any]] = []
    if positions.empty:
        return rows
    for variant, group in positions.groupby("variant"):
        forced_group = group[group["forced_unfilled_exit"].astype(bool)]
        rows.append(
            {
                "section": "aggregate",
                "metric": "forced_exit_rate",
                "variant": variant,
                "forced_exit_count": int(len(forced_group)),
                "position_count": int(len(group)),
                "value": _safe_ratio(len(forced_group), len(group)),
            }
        )
    if not forced.empty:
        for (variant, reason), group in forced.groupby(["variant", "failure_reason"]):
            rows.append({"section": "failure_reason_distribution", "variant": variant, "failure_reason": reason, "count": int(len(group))})
        forced_dates = forced.copy()
        forced_dates["expiry_dt"] = pd.to_datetime(forced_dates["expiry"], errors="coerce")
        forced_dates["entry_dte_num"] = pd.to_numeric(forced_dates["entry_dte"], errors="coerce")
        forced_dates["moneyness_bucket"] = pd.cut(pd.to_numeric(forced_dates["long_put_moneyness"], errors="coerce"), bins=[0, 0.85, 0.90, 0.93, 1.0], labels=["lt_0_85", "0_85_0_90", "0_90_0_93", "gt_0_93"])
        forced_dates["entry_dte_bucket"] = pd.cut(forced_dates["entry_dte_num"], bins=[0, 60, 90, 120, 10_000], labels=["lte_60", "61_90", "91_120", "gt_120"])
        for (variant, year), group in forced_dates.groupby(["variant", forced_dates["expiry_dt"].dt.year]):
            rows.append({"section": "forced_exits_by_year", "variant": variant, "year": int(year), "count": int(len(group))})
        for (variant, month), group in forced_dates.groupby(["variant", forced_dates["expiry_dt"].dt.month]):
            rows.append({"section": "forced_exits_by_expiry_month", "variant": variant, "expiry_month": int(month), "count": int(len(group))})
        for (variant, bucket), group in forced_dates.groupby(["variant", "moneyness_bucket"], observed=True):
            rows.append({"section": "forced_exits_by_moneyness_bucket", "variant": variant, "moneyness_bucket": str(bucket), "count": int(len(group))})
        for (variant, bucket), group in forced_dates.groupby(["variant", "entry_dte_bucket"], observed=True):
            rows.append({"section": "forced_exits_by_entry_dte", "variant": variant, "entry_dte_bucket": str(bucket), "count": int(len(group))})
    if not window.empty:
        liquid = window[window["both_legs_liquid"].astype(bool)].copy()
        if not liquid.empty:
            liquid["days_before_expiry"] = pd.to_numeric(liquid["dte"], errors="coerce")
            last_valid = liquid.groupby(["variant", "position_id"])["days_before_expiry"].min().reset_index()
            for variant, group in last_valid.groupby("variant"):
                rows.append({"section": "last_valid_quote_summary", "variant": variant, "metric": "average_last_valid_quote_days_before_expiry", "value": float(group["days_before_expiry"].mean())})
                rows.append({"section": "last_valid_quote_summary", "variant": variant, "metric": "median_last_valid_quote_days_before_expiry", "value": float(group["days_before_expiry"].median())})
        else:
            for variant in positions["variant"].unique():
                rows.append({"section": "last_valid_quote_summary", "variant": variant, "metric": "average_last_valid_quote_days_before_expiry", "value": ""})
                rows.append({"section": "last_valid_quote_summary", "variant": variant, "metric": "median_last_valid_quote_days_before_expiry", "value": ""})
    return rows


def _normal_vs_forced_comparison_rows(all_positions: list[dict[str, Any]], options: pd.DataFrame, config: dict) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pos in all_positions:
        window = pd.DataFrame(_exit_window_availability_rows(pos, options, config))
        if window.empty:
            continue
        rows.append(
            {
                "section": "exit_window_comparison",
                "variant": pos["variant"],
                "position_status": "forced" if pos["forced_unfilled_exit"] else "normal",
                "position_id": pos["position_id"],
                "average_volume": _mean_pair(window, "volume"),
                "average_open_interest": _mean_pair(window, "open_interest"),
                "average_spread_pct": _mean_pair(window, "spread_pct"),
                "liquid_day_count": int(window["both_legs_liquid"].astype(bool).sum()),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return rows
    for (variant, status), group in frame.groupby(["variant", "position_status"]):
        rows.append(
            {
                "section": "exit_window_comparison_summary",
                "variant": variant,
                "position_status": status,
                "position_count": int(len(group)),
                "average_volume": float(pd.to_numeric(group["average_volume"], errors="coerce").mean()),
                "average_open_interest": float(pd.to_numeric(group["average_open_interest"], errors="coerce").mean()),
                "average_spread_pct": float(pd.to_numeric(group["average_spread_pct"], errors="coerce").mean()),
                "average_liquid_day_count": float(pd.to_numeric(group["liquid_day_count"], errors="coerce").mean()),
            }
        )
    return rows


def _forced_markdown(analysis: pd.DataFrame) -> str:
    forced_rate = analysis[analysis["section"] == "aggregate"] if not analysis.empty else pd.DataFrame()
    reasons = analysis[analysis["section"] == "failure_reason_distribution"] if not analysis.empty else pd.DataFrame()
    last_valid = analysis[analysis["section"] == "last_valid_quote_summary"] if not analysis.empty else pd.DataFrame()
    comparison = analysis[analysis["section"] == "exit_window_comparison_summary"] if not analysis.empty else pd.DataFrame()
    lines = [
        "# Forced Exit Liquidity Analysis",
        "",
        "This report diagnoses forced_unfilled_exit positions only. It does not change exits, fills, quote gates, data, or strategy parameters.",
        "",
        "## Forced Exit Rate",
        "",
    ]
    if forced_rate.empty:
        lines.append("- No forced exit aggregate rows.")
    else:
        for row in forced_rate.itertuples(index=False):
            lines.append(f"- {row.variant}: {row.forced_exit_count}/{row.position_count} = {row.value}")
    lines.extend(["", "## Failure Reasons", ""])
    if reasons.empty:
        lines.append("- None.")
    else:
        for row in reasons.itertuples(index=False):
            lines.append(f"- {row.variant}: {row.failure_reason} = {row.count}")
    lines.extend(["", "## Last Valid Quote Days Before Expiry", ""])
    if last_valid.empty:
        lines.append("- Unavailable.")
    else:
        for row in last_valid.itertuples(index=False):
            lines.append(f"- {row.variant}: {row.metric} = {row.value}")
    lines.extend(["", "## Forced vs Normal Exit Window", ""])
    if comparison.empty:
        lines.append("- Unavailable.")
    else:
        for row in comparison.itertuples(index=False):
            lines.append(
                f"- {row.variant}/{row.position_status}: positions={row.position_count}, "
                f"avg_volume={row.average_volume}, avg_oi={row.average_open_interest}, "
                f"avg_spread={row.average_spread_pct}, avg_liquid_days={row.average_liquid_day_count}"
            )
    lines.extend(
        [
            "",
            "## Required Limitations",
            "",
            "- The report identifies whether forced exits are mainly data, quote quality, liquidity, DTE-window, or expiry-data driven.",
            "- Current results are not safe to interpret as final performance while forced_unfilled_exit remains high.",
            "- Remaining data quality limitations include VIX proxy usage, missing/invalid quote days, and incomplete tradable exit windows.",
            "- No parameter changes or performance interpretation are recommended here.",
        ]
    )
    return "\n".join(lines) + "\n"


def _entry_exit_feasibility_analysis(
    variant_runs: list[dict[str, Any]],
    options: pd.DataFrame,
    config: dict,
    put_params: dict,
) -> pd.DataFrame:
    opt = options.copy()
    if not opt.empty:
        opt["date"] = pd.to_datetime(opt["date"], errors="coerce")
        opt["expiry"] = pd.to_datetime(opt["expiry"], errors="coerce")
    rows: list[dict[str, Any]] = []
    position_rows: list[dict[str, Any]] = []
    for run in variant_runs:
        variant = str(run["variant"])
        positions = _position_metadata(variant, run["trades"].copy(), run["lifecycle"].copy())
        for pos in positions:
            feasibility = _position_entry_exit_feasibility(pos, opt, config, put_params)
            position_row = {**pos, **feasibility, "section": "position_feasibility"}
            position_rows.append(position_row)
            rows.append(position_row)
    rows.extend(_feasibility_aggregate_rows(pd.DataFrame(position_rows)))
    return pd.DataFrame(rows)


def _position_entry_exit_feasibility(pos: dict[str, Any], options: pd.DataFrame, config: dict, put_params: dict) -> dict[str, Any]:
    entry_date = pd.Timestamp(pos["entry_date"])
    expiry = pd.Timestamp(pos["expiry"])
    long_strike = float(pos["long_put_strike"])
    short_strike = float(pos["short_put_strike"])
    chain = options[(options["date"] == entry_date) & (options["expiry"] == expiry) & (options["cp"] == "P")].copy()
    long_entry = _leg_quote(chain, entry_date, long_strike)
    short_entry = _leg_quote(chain, entry_date, short_strike)
    long_rank = _nearby_rank(chain, long_strike)
    short_rank = _nearby_rank(chain, short_strike)
    long_decay = _leg_decay_profile(options, entry_date, expiry, long_strike, config, put_params)
    short_decay = _leg_decay_profile(options, entry_date, expiry, short_strike, config, put_params)
    valid_ratio = min(_num(long_decay["valid_quote_ratio"]), _num(short_decay["valid_quote_ratio"]))
    tradable_days = min(int(long_decay["tradable_days_count"]), int(short_decay["tradable_days_count"]))
    median_oi = min(_num(long_decay["median_open_interest_until_dte14"]), _num(short_decay["median_open_interest_until_dte14"]))
    feasibility_class = _feasibility_class(valid_ratio, tradable_days, median_oi, config)
    predictable = bool(pos["forced_unfilled_exit"] and feasibility_class != "EXIT_FEASIBLE")
    return {
        "long_leg_entry_quote_quality_status": _quote_value(long_entry, "quote_quality_status"),
        "short_leg_entry_quote_quality_status": _quote_value(short_entry, "quote_quality_status"),
        "long_leg_entry_spread_pct": _quote_float(long_entry, "spread_pct"),
        "short_leg_entry_spread_pct": _quote_float(short_entry, "spread_pct"),
        "long_leg_entry_volume": _quote_float(long_entry, "volume"),
        "short_leg_entry_volume": _quote_float(short_entry, "volume"),
        "long_leg_entry_open_interest": _quote_float(long_entry, "open_interest"),
        "short_leg_entry_open_interest": _quote_float(short_entry, "open_interest"),
        "chosen_long_put_volume_rank": long_rank["volume_rank"],
        "chosen_long_put_oi_rank": long_rank["oi_rank"],
        "chosen_long_put_spread_rank": long_rank["spread_rank"],
        "chosen_short_put_volume_rank": short_rank["volume_rank"],
        "chosen_short_put_oi_rank": short_rank["oi_rank"],
        "chosen_short_put_spread_rank": short_rank["spread_rank"],
        "nearby_more_liquid_long_put_exists": long_rank["nearby_more_liquid_exists"],
        "nearby_more_liquid_short_put_exists": short_rank["nearby_more_liquid_exists"],
        **{f"long_leg_{key}": value for key, value in long_decay.items()},
        **{f"short_leg_{key}": value for key, value in short_decay.items()},
        "combined_valid_quote_ratio": valid_ratio,
        "combined_tradable_days_count": tradable_days,
        "combined_median_open_interest_until_dte14": median_oi,
        "feasibility_class": feasibility_class,
        "forced_exit_predictable_at_entry": predictable,
    }


def _nearby_rank(chain: pd.DataFrame, strike: float) -> dict[str, Any]:
    if chain.empty:
        return {"volume_rank": "", "oi_rank": "", "spread_rank": "", "nearby_more_liquid_exists": ""}
    strikes = sorted(chain["strike"].dropna().unique())
    if not strikes:
        return {"volume_rank": "", "oi_rank": "", "spread_rank": "", "nearby_more_liquid_exists": ""}
    closest_index = min(range(len(strikes)), key=lambda idx: abs(float(strikes[idx]) - float(strike)))
    nearby_strikes = set(strikes[max(0, closest_index - 5) : closest_index + 6])
    nearby = chain[chain["strike"].isin(nearby_strikes)].copy()
    chosen = nearby[np.isclose(nearby["strike"], float(strike))]
    if nearby.empty or chosen.empty:
        return {"volume_rank": "", "oi_rank": "", "spread_rank": "", "nearby_more_liquid_exists": ""}
    row = chosen.iloc[0]
    nearby["volume_rank"] = pd.to_numeric(nearby["volume"], errors="coerce").rank(method="min", ascending=False)
    nearby["oi_rank"] = pd.to_numeric(nearby["open_interest"], errors="coerce").rank(method="min", ascending=False)
    nearby["spread_rank"] = pd.to_numeric(nearby["spread_pct"], errors="coerce").rank(method="min", ascending=True)
    ranked = nearby[np.isclose(nearby["strike"], float(strike))].iloc[0]
    chosen_volume = float(row.get("volume", 0.0))
    chosen_oi = float(row.get("open_interest", 0.0))
    chosen_spread = _num(row.get("spread_pct"))
    more_liquid = nearby[
        (pd.to_numeric(nearby["volume"], errors="coerce") > chosen_volume)
        & (pd.to_numeric(nearby["open_interest"], errors="coerce") >= chosen_oi)
        & (pd.to_numeric(nearby["spread_pct"], errors="coerce") <= chosen_spread)
    ]
    return {
        "volume_rank": int(ranked["volume_rank"]) if pd.notna(ranked["volume_rank"]) else "",
        "oi_rank": int(ranked["oi_rank"]) if pd.notna(ranked["oi_rank"]) else "",
        "spread_rank": int(ranked["spread_rank"]) if pd.notna(ranked["spread_rank"]) else "",
        "nearby_more_liquid_exists": bool(not more_liquid.empty),
    }


def _leg_decay_profile(
    options: pd.DataFrame,
    entry_date: pd.Timestamp,
    expiry: pd.Timestamp,
    strike: float,
    config: dict,
    put_params: dict,
) -> dict[str, Any]:
    end_date = expiry - pd.Timedelta(days=int(put_params.get("exit_dte", 14)))
    leg = options[
        (options["date"] >= entry_date)
        & (options["date"] <= end_date)
        & (options["expiry"] == expiry)
        & (options["cp"] == "P")
        & np.isclose(options["strike"], float(strike))
    ].copy()
    if leg.empty:
        return {
            "tradable_days_count": 0,
            "non_tradable_days_count": 0,
            "valid_quote_ratio": 0.0,
            "first_non_valid_quote_date": "",
            "last_valid_quote_date": "",
            "days_from_last_valid_quote_to_expiry": "",
            "average_spread_pct_until_dte14": "",
            "median_volume_until_dte14": "",
            "median_open_interest_until_dte14": "",
        }
    valid = _valid_quote_mask(leg, config)
    first_bad = leg.loc[~valid, "date"].min() if (~valid).any() else pd.NaT
    last_valid = leg.loc[valid, "date"].max() if valid.any() else pd.NaT
    return {
        "tradable_days_count": int(valid.sum()),
        "non_tradable_days_count": int((~valid).sum()),
        "valid_quote_ratio": float(valid.mean()) if len(valid) else 0.0,
        "first_non_valid_quote_date": "" if pd.isna(first_bad) else str(pd.Timestamp(first_bad).date()),
        "last_valid_quote_date": "" if pd.isna(last_valid) else str(pd.Timestamp(last_valid).date()),
        "days_from_last_valid_quote_to_expiry": "" if pd.isna(last_valid) else int((expiry - pd.Timestamp(last_valid)).days),
        "average_spread_pct_until_dte14": _float_or_blank(pd.to_numeric(leg["spread_pct"], errors="coerce").mean()),
        "median_volume_until_dte14": _float_or_blank(pd.to_numeric(leg["volume"], errors="coerce").median()),
        "median_open_interest_until_dte14": _float_or_blank(pd.to_numeric(leg["open_interest"], errors="coerce").median()),
    }


def _valid_quote_mask(frame: pd.DataFrame, config: dict) -> pd.Series:
    min_volume = float(config.get("wide_spread_volume_threshold", 50))
    min_oi = float(config.get("min_open_interest", 100))
    return (
        frame["quote_quality_status"].astype(str).eq("VALID")
        & frame["is_tradable_quote"].astype(bool)
        & (pd.to_numeric(frame["volume"], errors="coerce") >= min_volume)
        & (pd.to_numeric(frame["open_interest"], errors="coerce") >= min_oi)
    )


def _feasibility_class(valid_ratio: float, tradable_days: int, median_oi: float, config: dict) -> str:
    if valid_ratio >= 0.6 and tradable_days >= 10 and median_oi >= float(config.get("min_open_interest", 100)):
        return "EXIT_FEASIBLE"
    if valid_ratio >= 0.3 or tradable_days >= 3:
        return "EXIT_FRAGILE"
    return "EXIT_UNLIKELY"


def _feasibility_aggregate_rows(position_rows: pd.DataFrame) -> list[dict[str, Any]]:
    if position_rows.empty:
        return []
    rows: list[dict[str, Any]] = []
    for (variant, klass), group in position_rows.groupby(["variant", "feasibility_class"]):
        rows.append({"section": "feasibility_distribution", "variant": variant, "feasibility_class": klass, "count": int(len(group))})
    for (variant, klass), group in position_rows.groupby(["variant", "feasibility_class"]):
        forced = int(group["forced_unfilled_exit"].astype(bool).sum())
        rows.append(
            {
                "section": "forced_exit_rate_by_feasibility",
                "variant": variant,
                "feasibility_class": klass,
                "forced_exit_count": forced,
                "position_count": int(len(group)),
                "value": _safe_ratio(forced, len(group)),
            }
        )
    forced = position_rows[position_rows["forced_unfilled_exit"].astype(bool)].copy()
    if not forced.empty:
        forced["long_rank_bucket"] = pd.cut(pd.to_numeric(forced["chosen_long_put_volume_rank"], errors="coerce"), bins=[0, 1, 3, 10_000], labels=["rank_1", "rank_2_3", "rank_gt_3"])
        forced["long_moneyness_bucket"] = pd.cut(pd.to_numeric(forced["long_put_moneyness"], errors="coerce"), bins=[0, 0.85, 0.90, 0.93, 1.0], labels=["lt_0_85", "0_85_0_90", "0_90_0_93", "gt_0_93"])
        forced["entry_oi_bucket"] = pd.cut(pd.to_numeric(forced["long_leg_entry_open_interest"], errors="coerce"), bins=[0, 100, 500, 2000, 10_000_000], labels=["lte_100", "101_500", "501_2000", "gt_2000"])
        forced["entry_volume_bucket"] = pd.cut(pd.to_numeric(forced["long_leg_entry_volume"], errors="coerce"), bins=[0, 50, 200, 1000, 10_000_000], labels=["lte_50", "51_200", "201_1000", "gt_1000"])
        for (variant, bucket), group in forced.groupby(["variant", "long_rank_bucket"], observed=True):
            rows.append({"section": "forced_exits_by_chosen_long_put_liquidity_rank", "variant": variant, "rank_bucket": str(bucket), "count": int(len(group))})
        for (variant, bucket), group in forced.groupby(["variant", "long_moneyness_bucket"], observed=True):
            rows.append({"section": "forced_exits_by_chosen_long_put_moneyness", "variant": variant, "moneyness_bucket": str(bucket), "count": int(len(group))})
        for (variant, bucket), group in forced.groupby(["variant", "entry_oi_bucket"], observed=True):
            rows.append({"section": "forced_exits_by_entry_open_interest_bucket", "variant": variant, "entry_oi_bucket": str(bucket), "count": int(len(group))})
        for (variant, bucket), group in forced.groupby(["variant", "entry_volume_bucket"], observed=True):
            rows.append({"section": "forced_exits_by_entry_volume_bucket", "variant": variant, "entry_volume_bucket": str(bucket), "count": int(len(group))})
    for variant, group in position_rows.groupby("variant"):
        forced_group = group[group["forced_unfilled_exit"].astype(bool)]
        predictable = int(forced_group["forced_exit_predictable_at_entry"].astype(bool).sum()) if not forced_group.empty else 0
        more_liquid = int(forced_group["nearby_more_liquid_long_put_exists"].astype(bool).sum()) if not forced_group.empty else 0
        rows.append(
            {
                "section": "entry_predictability_summary",
                "variant": variant,
                "forced_exit_count": int(len(forced_group)),
                "predictable_at_entry_count": predictable,
                "predictable_at_entry_ratio": _safe_ratio(predictable, len(forced_group)),
                "forced_with_more_liquid_long_nearby_count": more_liquid,
                "forced_with_more_liquid_long_nearby_ratio": _safe_ratio(more_liquid, len(forced_group)),
            }
        )
    return rows


def _feasibility_markdown(analysis: pd.DataFrame) -> str:
    dist = analysis[analysis["section"] == "feasibility_distribution"] if not analysis.empty else pd.DataFrame()
    rates = analysis[analysis["section"] == "forced_exit_rate_by_feasibility"] if not analysis.empty else pd.DataFrame()
    predict = analysis[analysis["section"] == "entry_predictability_summary"] if not analysis.empty else pd.DataFrame()
    lines = [
        "# Entry Exit Feasibility Analysis",
        "",
        "This report diagnoses whether exit fragility was visible at entry time. It does not change selectors, exits, quote gates, fills, or strategy parameters.",
        "",
        "## Feasibility Distribution",
        "",
    ]
    if dist.empty:
        lines.append("- No feasibility rows.")
    else:
        for row in dist.itertuples(index=False):
            lines.append(f"- {row.variant}: {row.feasibility_class} = {row.count}")
    lines.extend(["", "## Forced Exit Rate By Feasibility", ""])
    if rates.empty:
        lines.append("- No forced exit rate rows.")
    else:
        for row in rates.itertuples(index=False):
            lines.append(f"- {row.variant}/{row.feasibility_class}: {row.forced_exit_count}/{row.position_count} = {row.value}")
    lines.extend(["", "## Entry-Time Predictability", ""])
    if predict.empty:
        lines.append("- No predictability rows.")
    else:
        for row in predict.itertuples(index=False):
            lines.append(
                f"- {row.variant}: predictable_at_entry={row.predictable_at_entry_count}/{row.forced_exit_count} "
                f"({row.predictable_at_entry_ratio}), more_liquid_long_nearby={row.forced_with_more_liquid_long_nearby_count}/{row.forced_exit_count} "
                f"({row.forced_with_more_liquid_long_nearby_ratio})"
            )
    lines.extend(
        [
            "",
            "## Required Limitations",
            "",
            "- This report answers whether forced exits were visible from entry-time and historical quote behavior.",
            "- It does not recommend parameter changes.",
            "- It does not exclude trades or modify the selector.",
            "- It can indicate whether a future execution feasibility filter may be needed, but no filter is implemented here.",
        ]
    )
    return "\n".join(lines) + "\n"


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


def _leg_quote(relevant: pd.DataFrame, date: pd.Timestamp, strike: float) -> dict[str, Any] | None:
    rows = relevant[(relevant["date"] == date) & np.isclose(relevant["strike"], float(strike))]
    if rows.empty:
        return None
    return rows.iloc[0].to_dict()


def _quote_value(quote: dict[str, Any] | None, key: str) -> Any:
    return "" if quote is None else quote.get(key, "")


def _quote_bool(quote: dict[str, Any] | None, key: str) -> bool | str:
    if quote is None:
        return ""
    value = quote.get(key, False)
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"true", "1", "yes", "y"}


def _quote_float(quote: dict[str, Any] | None, key: str) -> float | str:
    if quote is None:
        return ""
    return _float_or_blank(quote.get(key))


def _both_liquid(long_quote: dict[str, Any] | None, short_quote: dict[str, Any] | None, config: dict) -> bool:
    if long_quote is None or short_quote is None:
        return False
    min_volume = float(config.get("wide_spread_volume_threshold", 50))
    min_oi = float(config.get("min_open_interest", 100))
    return all(
        [
            str(long_quote.get("quote_quality_status", "")) == "VALID",
            str(short_quote.get("quote_quality_status", "")) == "VALID",
            _quote_bool(long_quote, "is_tradable_quote") is True,
            _quote_bool(short_quote, "is_tradable_quote") is True,
            float(long_quote.get("volume", 0.0)) >= min_volume,
            float(short_quote.get("volume", 0.0)) >= min_volume,
            float(long_quote.get("open_interest", 0.0)) >= min_oi,
            float(short_quote.get("open_interest", 0.0)) >= min_oi,
        ]
    )


def _has_non_valid(window: pd.DataFrame, side: str) -> bool:
    exists = window[f"{side}_leg_quote_exists"].astype(bool)
    if not exists.any():
        return False
    status = window.loc[exists, f"{side}_leg_quote_quality_status"].astype(str)
    tradable = window.loc[exists, f"{side}_leg_is_tradable_quote"].astype(str).str.lower().isin({"true", "1", "yes", "y"})
    return bool((status != "VALID").any() or (~tradable).any())


def _mean_pair(window: pd.DataFrame, suffix: str) -> float | str:
    cols = [f"long_leg_{suffix}", f"short_leg_{suffix}"]
    values = []
    for col in cols:
        if col in window:
            values.extend(pd.to_numeric(window[col], errors="coerce").dropna().tolist())
    if not values:
        return ""
    return float(np.mean(values))


def _first_numeric(frame: pd.DataFrame, columns: list[str]) -> float:
    for col in columns:
        if col in frame:
            values = pd.to_numeric(frame[col], errors="coerce").dropna()
            if not values.empty:
                return float(values.iloc[0])
    return np.nan


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


def _float_or_blank(value: Any) -> float | str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(value):
        return ""
    return value


def _num(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _safe_ratio(numerator: float, denominator: float) -> float | str:
    if pd.isna(denominator) or denominator == 0:
        return ""
    return float(numerator / denominator)
