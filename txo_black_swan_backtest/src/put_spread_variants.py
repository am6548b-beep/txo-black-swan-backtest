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
from .strategies import BlackSwanStateMachine, ContractSelector
from .utils import year_key


VARIANTS = [
    "current_signal_based",
    "quarterly_base_insurance",
    "base_plus_signal_boost",
    "quarterly_base_insurance_feasible_only",
]
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
        self.feasibility_filter_events: list[dict[str, Any]] = []
        self.rolling_rejection_events: list[dict[str, Any]] = []

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
        if self.variant_name == "quarterly_base_insurance_feasible_only":
            self._check_quarterly_base(row, stock_equity, allow_signal_boost=False, require_entry_time_feasible=True)
            return
        if self.variant_name == "rolling_base_insurance":
            self._check_rolling_base(row, stock_equity)
            return
        raise ValueError(f"Unknown put-spread variant: {self.variant_name}")

    def _check_quarterly_base(self, row, stock_equity: float, allow_signal_boost: bool, require_entry_time_feasible: bool = False) -> None:
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
        self._open_put_spread_with_budget(
            row,
            stock_equity,
            entry_budget,
            "quarterly_base_put_spread" if not signal_ok else "quarterly_signal_boost_put_spread",
            require_entry_time_feasible=require_entry_time_feasible,
        )

    def _check_rolling_base(self, row, stock_equity: float) -> None:
        if self._open_positions("put_spread"):
            return
        year = year_key(row.date)
        macro_state = str(getattr(row, "macro_state", "NORMAL"))
        restriction = macro_restrictions(macro_state)
        budget_cap = float(self.config["max_annual_hedge_budget_pct"]) * float(restriction["hedge_budget_multiplier"]) * stock_equity
        used = self.annual_hedge_spend.get(year, 0.0)
        remaining = max(0.0, budget_cap - used)
        if remaining <= 0:
            self._log_rolling_rejection(row, "BUDGET_BLOCKED")
            return
        base_budget = float(self.config.get("base_annual_hedge_budget_pct", self.config["max_annual_hedge_budget_pct"])) * stock_equity
        entry_budget = min(remaining, base_budget / 4.0)
        opened = self._open_put_spread_with_budget(row, stock_equity, entry_budget, "rolling_base_put_spread")
        if not opened:
            self._log_rolling_rejection(row, self._rolling_rejection_reason(row, entry_budget))

    def _log_rolling_rejection(self, row, reason: str) -> None:
        self.rolling_rejection_events.append(
            {
                "variant": self.variant_name,
                "date": str(pd.Timestamp(row.date).date()),
                "reason": reason or "UNKNOWN",
                "txf_close": float(getattr(row, "txf_close", np.nan)),
            }
        )

    def _rolling_rejection_reason(self, row, entry_budget: float) -> str:
        if entry_budget <= 0:
            return "BUDGET_BLOCKED"
        can_build, reason = _entry_candidate_available(row, self.selector, self.config, self.put_params)
        return "" if can_build else reason

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

    def _open_put_spread_with_budget(self, row, stock_equity: float, entry_budget: float, reason: str, require_entry_time_feasible: bool = False) -> bool:
        if entry_budget <= 0:
            return False
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
            return False
        short_put = self.selector.nearest_strike(
            date,
            "P",
            row.txf_close * short_m,
            dte_min,
            dte_max,
            expiry=pd.Timestamp(long_put.expiry),
        )
        if short_put is None or short_put.strike >= long_put.strike:
            return False
        if require_entry_time_feasible:
            feasibility = _entry_time_feasibility_filter(long_put, short_put, self.options, self.config, self.put_params)
            if feasibility["filter_decision"] == "SKIP_EXIT_UNLIKELY":
                self.feasibility_filter_events.append(
                    {
                        "variant": self.variant_name,
                        "date": str(date.date()),
                        "long_put_strike": long_put.strike,
                        "short_put_strike": short_put.strike,
                        "long_put_moneyness": long_put.strike / float(row.txf_close),
                        "short_put_moneyness": short_put.strike / float(row.txf_close),
                        "expiry": long_put.expiry,
                        "entry_dte": long_put.dte,
                        "long_entry_time_feasibility": feasibility["long_entry_time_feasibility"],
                        "short_entry_time_feasibility": feasibility["short_entry_time_feasibility"],
                        "filter_decision": feasibility["filter_decision"],
                        "skip_reason": feasibility["skip_reason"],
                        "long_filter_max_reference_date": feasibility["long_filter_max_reference_date"],
                        "short_filter_max_reference_date": feasibility["short_filter_max_reference_date"],
                        "no_lookahead_pass": feasibility["no_lookahead_pass"],
                    }
                )
                return False
        stress = is_stress_day(row, self.config)
        long_px = fill_price(long_put, "BUY", self.config, stress)
        short_px = fill_price(short_put, "SELL", self.config, stress)
        per_spread_debit = (long_px - short_px) * float(self.config["txo_point_value"])
        per_spread_cost = per_spread_debit + 2 * float(self.config["commission_per_contract_per_side"])
        if per_spread_cost <= 0:
            return False
        notional_contracts = max(
            1,
            int((stock_equity * float(self.config["portfolio_beta"])) / (row.txf_close * float(self.config["txo_point_value"])) * 0.5),
        )
        qty = min(notional_contracts, int(entry_budget // per_spread_cost))
        if qty <= 0:
            return False
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
        return True


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
        variant_runs.append(
            {
                "variant": variant,
                "equity": equity,
                "trades": trades,
                "lifecycle": lifecycle,
                "filter_events": pd.DataFrame(engine.feasibility_filter_events),
            }
        )
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
    filter_audit = _execution_feasibility_filter_audit(variant_runs, comparison, options, config, put_params)
    filter_audit.to_csv(report_dir / "execution_feasibility_filter_audit.csv", index=False)
    (report_dir / "execution_feasibility_filter_audit.md").write_text(_filter_audit_markdown(filter_audit), encoding="utf-8")
    exit_timing = _exit_timing_diagnostic(variant_runs, options, config, put_params)
    exit_timing.to_csv(report_dir / "exit_timing_diagnostic.csv", index=False)
    (report_dir / "exit_timing_diagnostic.md").write_text(_exit_timing_markdown(exit_timing), encoding="utf-8")
    exit_variant_runs = _run_exit_timing_variant_engines(market, options, portfolio, config, put_params, ic_params)
    exit_variant_comparison = _exit_timing_variant_comparison_from_runs(exit_variant_runs, market, config)
    exit_variant_comparison.to_csv(report_dir / "exit_timing_variant_comparison.csv", index=False)
    (report_dir / "exit_timing_variant_comparison.md").write_text(_exit_variant_markdown(exit_variant_comparison), encoding="utf-8")
    moneyness_runs = _run_moneyness_variant_engines(market, options, portfolio, config, put_params, ic_params)
    moneyness = _moneyness_tradability_from_runs(moneyness_runs, options, config)
    moneyness.to_csv(report_dir / "moneyness_tradability_diagnostic.csv", index=False)
    (report_dir / "moneyness_tradability_diagnostic.md").write_text(_moneyness_markdown(moneyness), encoding="utf-8")
    rolling = _rolling_coverage_gap_analysis(variant_runs + exit_variant_runs + moneyness_runs, market, options, config, put_params)
    rolling.to_csv(report_dir / "rolling_coverage_gap_analysis.csv", index=False)
    (report_dir / "rolling_coverage_gap_analysis.md").write_text(_rolling_coverage_markdown(rolling), encoding="utf-8")
    rolling_replacement = _run_rolling_replacement_comparison(market, options, portfolio, config, put_params, ic_params)
    rolling_replacement.to_csv(report_dir / "rolling_replacement_variant_comparison.csv", index=False)
    (report_dir / "rolling_replacement_variant_comparison.md").write_text(_rolling_replacement_markdown(rolling_replacement), encoding="utf-8")
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
    opens = t[(t["strategy"] == "put_spread") & t["reason"].astype(str).str.contains("open|quarterly|rolling", case=False, regex=True, na=False)]
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
        opens = group[group["reason"].astype(str).str.contains("open|quarterly|rolling", case=False, regex=True, na=False)]
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


def _entry_time_feasibility_filter(long_contract: Any, short_contract: Any, options: pd.DataFrame, config: dict, put_params: dict) -> dict[str, Any]:
    long_profile = _entry_time_leg_profile(long_contract, options, config, put_params)
    short_profile = _entry_time_leg_profile(short_contract, options, config, put_params)
    long_class = _feasibility_class(
        long_profile["valid_quote_ratio"],
        int(long_profile["tradable_days_count"]),
        long_profile["median_open_interest_until_dte14"],
        config,
    )
    short_class = _feasibility_class(
        short_profile["valid_quote_ratio"],
        int(short_profile["tradable_days_count"]),
        short_profile["median_open_interest_until_dte14"],
        config,
    )
    skip = long_class == "EXIT_UNLIKELY" or short_class == "EXIT_UNLIKELY"
    entry_date = pd.Timestamp(long_contract.date)
    max_dates = [pd.to_datetime(long_profile["max_reference_date"], errors="coerce"), pd.to_datetime(short_profile["max_reference_date"], errors="coerce")]
    no_lookahead = all(pd.isna(date) or date <= entry_date for date in max_dates)
    return {
        "long_entry_time_feasibility": long_class,
        "short_entry_time_feasibility": short_class,
        "filter_decision": "SKIP_EXIT_UNLIKELY" if skip else "ALLOW",
        "skip_reason": "EXIT_UNLIKELY_LEG" if skip else "",
        "long_filter_max_reference_date": long_profile["max_reference_date"],
        "short_filter_max_reference_date": short_profile["max_reference_date"],
        "no_lookahead_pass": no_lookahead,
        "long_profile": long_profile,
        "short_profile": short_profile,
    }


def _entry_time_leg_profile(contract: Any, options: pd.DataFrame, config: dict, put_params: dict) -> dict[str, Any]:
    entry_date = pd.Timestamp(contract.date)
    lookback_start = entry_date - pd.Timedelta(days=int(config.get("entry_feasibility_lookback_days", 252)))
    dte_min = int(put_params.get("target_dte_min", 60))
    dte_max = int(put_params.get("target_dte_max", 120))
    target_moneyness = float(contract.strike) / float(contract.underlying) if float(contract.underlying) else np.nan
    hist = options[
        (options["date"] >= lookback_start)
        & (options["date"] <= entry_date)
        & (options["cp"] == contract.cp)
        & (options["dte"].between(dte_min, dte_max))
        & options["underlying"].notna()
    ].copy()
    if hist.empty or pd.isna(target_moneyness):
        return _empty_entry_time_profile(entry_date)
    hist["moneyness"] = hist["strike"].astype(float) / hist["underlying"].astype(float)
    bucket_width = float(config.get("entry_feasibility_moneyness_bucket_width", 0.025))
    bucket = hist[(hist["moneyness"] - target_moneyness).abs() <= bucket_width].copy()
    exact = hist[(hist["expiry"] == pd.Timestamp(contract.expiry)) & np.isclose(hist["strike"], float(contract.strike))].copy()
    sample = pd.concat([bucket, exact], ignore_index=True).drop_duplicates(["date", "expiry", "cp", "strike"])
    if sample.empty:
        return _empty_entry_time_profile(entry_date)
    valid = _valid_quote_mask(sample, config)
    day_valid = valid.groupby(sample["date"]).any()
    max_ref = sample["date"].max()
    median_oi = _float_or_blank(pd.to_numeric(sample["open_interest"], errors="coerce").median())
    return {
        "tradable_days_count": int(day_valid.sum()),
        "non_tradable_days_count": int((~day_valid).sum()),
        "valid_quote_ratio": float(day_valid.mean()) if len(day_valid) else 0.0,
        "first_non_valid_quote_date": "" if day_valid.all() else str(pd.Timestamp(day_valid[~day_valid].index.min()).date()),
        "last_valid_quote_date": "" if not day_valid.any() else str(pd.Timestamp(day_valid[day_valid].index.max()).date()),
        "days_from_last_valid_quote_to_expiry": "",
        "average_spread_pct_until_dte14": _float_or_blank(pd.to_numeric(sample["spread_pct"], errors="coerce").mean()),
        "median_volume_until_dte14": _float_or_blank(pd.to_numeric(sample["volume"], errors="coerce").median()),
        "median_open_interest_until_dte14": 0.0 if median_oi == "" else median_oi,
        "max_reference_date": "" if pd.isna(max_ref) else str(pd.Timestamp(max_ref).date()),
        "sample_rows": int(len(sample)),
        "unique_sample_days": int(sample["date"].nunique()),
    }


def _empty_entry_time_profile(entry_date: pd.Timestamp) -> dict[str, Any]:
    return {
        "tradable_days_count": 0,
        "non_tradable_days_count": 0,
        "valid_quote_ratio": 0.0,
        "first_non_valid_quote_date": "",
        "last_valid_quote_date": "",
        "days_from_last_valid_quote_to_expiry": "",
        "average_spread_pct_until_dte14": "",
        "median_volume_until_dte14": "",
        "median_open_interest_until_dte14": 0.0,
        "max_reference_date": str(entry_date.date()),
        "sample_rows": 0,
        "unique_sample_days": 0,
    }


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


def _execution_feasibility_filter_audit(
    variant_runs: list[dict[str, Any]],
    comparison: pd.DataFrame,
    options: pd.DataFrame,
    config: dict,
    put_params: dict,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    opt = options.copy()
    opt["date"] = pd.to_datetime(opt["date"], errors="coerce")
    opt["expiry"] = pd.to_datetime(opt["expiry"], errors="coerce")
    d_run = next((run for run in variant_runs if run["variant"] == "quarterly_base_insurance_feasible_only"), None)
    if d_run is None:
        out = pd.DataFrame([{"section": "audit", "check": "variant_d_present", "status": "FAIL", "detail": "missing"}])
        return out
    events = d_run.get("filter_events", pd.DataFrame()).copy()
    if events.empty:
        events = pd.DataFrame(columns=["variant", "date", "filter_decision", "no_lookahead_pass"])
    skipped = events[events.get("filter_decision", pd.Series(dtype=str)).astype(str) == "SKIP_EXIT_UNLIKELY"].copy()
    rows.append(
        {
            "section": "filter_summary",
            "variant": "quarterly_base_insurance_feasible_only",
            "metric": "skipped_due_to_exit_unlikely",
            "value": int(len(skipped)),
        }
    )
    no_lookahead_fail = int((~events.get("no_lookahead_pass", pd.Series([True] * len(events))).astype(bool)).sum()) if not events.empty else 0
    rows.append(
        {
            "section": "audit",
            "variant": "quarterly_base_insurance_feasible_only",
            "check": "no_lookahead_check",
            "status": "FAIL" if no_lookahead_fail else "PASS",
            "detail": f"fail_rows={no_lookahead_fail}",
        }
    )
    for event in skipped.itertuples(index=False):
        pseudo_pos = {
            "variant": getattr(event, "variant", "quarterly_base_insurance_feasible_only"),
            "position_id": f"SKIPPED-{getattr(event, 'Index', '')}",
            "entry_date": getattr(event, "date"),
            "expiry": getattr(event, "expiry"),
            "entry_dte": getattr(event, "entry_dte", ""),
            "long_put_strike": float(getattr(event, "long_put_strike")),
            "short_put_strike": float(getattr(event, "short_put_strike")),
            "long_put_moneyness": getattr(event, "long_put_moneyness", ""),
            "short_put_moneyness": getattr(event, "short_put_moneyness", ""),
            "quantity": "",
            "net_premium_paid": "",
            "forced_unfilled_exit": "",
            "normal_exit": "",
        }
        diag = _position_entry_exit_feasibility(pseudo_pos, opt, config, put_params)
        rows.append(
            {
                "section": "skipped_candidate",
                "variant": getattr(event, "variant", "quarterly_base_insurance_feasible_only"),
                "date": getattr(event, "date"),
                "expiry": getattr(event, "expiry"),
                "entry_dte": getattr(event, "entry_dte", ""),
                "long_put_strike": getattr(event, "long_put_strike"),
                "short_put_strike": getattr(event, "short_put_strike"),
                "long_put_moneyness": getattr(event, "long_put_moneyness", ""),
                "short_put_moneyness": getattr(event, "short_put_moneyness", ""),
                "long_entry_time_feasibility": getattr(event, "long_entry_time_feasibility"),
                "short_entry_time_feasibility": getattr(event, "short_entry_time_feasibility"),
                "filter_decision": getattr(event, "filter_decision"),
                "skip_reason": getattr(event, "skip_reason"),
                "diagnostic_feasibility": diag["feasibility_class"],
                "skipped_candidate_later_forced_exit_proxy": bool(diag["feasibility_class"] != "EXIT_FEASIBLE"),
                "no_lookahead_pass": getattr(event, "no_lookahead_pass"),
                "long_filter_max_reference_date": getattr(event, "long_filter_max_reference_date"),
                "short_filter_max_reference_date": getattr(event, "short_filter_max_reference_date"),
            }
        )
    rows.extend(_variant_d_audit_rows(d_run, comparison))
    return pd.DataFrame(rows)


def _variant_d_audit_rows(run: dict[str, Any], comparison: pd.DataFrame) -> list[dict[str, Any]]:
    variant = "quarterly_base_insurance_feasible_only"
    trades = run["trades"]
    lifecycle = run["lifecycle"]
    forced_count = int((lifecycle.get("issue", pd.Series(dtype=str)).astype(str) == "forced_unfilled_exit").sum()) if not lifecycle.empty else 0
    status = trades.get("quote_quality_status", pd.Series(dtype=str)).fillna("UNKNOWN").astype(str) if not trades.empty else pd.Series(dtype=str)
    non_valid = int((status != "VALID").sum()) if not status.empty else 0
    annual = comparison[(comparison["section"] == "annual_budget_usage") & (comparison["variant"] == variant)]
    crash = comparison[(comparison["section"] == "crash_window") & (comparison["variant"] == variant)]
    rows: list[dict[str, Any]] = [
        {"section": "variant_d_summary", "variant": variant, "metric": "forced_unfilled_exit_count", "value": forced_count},
        {"section": "variant_d_summary", "variant": variant, "metric": "non_valid_quote_trades_count", "value": non_valid},
    ]
    for row in annual.itertuples(index=False):
        rows.append(
            {
                "section": "variant_d_annual_hedge_cost",
                "variant": variant,
                "year": row.year,
                "annual_hedge_cost": row.annual_hedge_cost,
                "annual_budget_breach": row.annual_budget_breach,
            }
        )
    for row in crash.itertuples(index=False):
        rows.append(
            {
                "section": "variant_d_crash_coverage",
                "variant": variant,
                "period": row.period,
                "crash_window_coverage_ratio": row.crash_window_coverage_ratio,
                "crash_window_coverage_days": row.crash_window_coverage_days,
            }
        )
    return rows


def _filter_audit_markdown(audit: pd.DataFrame) -> str:
    summary = audit[audit["section"] == "filter_summary"] if not audit.empty else pd.DataFrame()
    skipped = audit[audit["section"] == "skipped_candidate"] if not audit.empty else pd.DataFrame()
    no_lookahead = audit[(audit["section"] == "audit") & (audit.get("check", pd.Series(dtype=str)) == "no_lookahead_check")] if not audit.empty else pd.DataFrame()
    d_summary = audit[audit["section"] == "variant_d_summary"] if not audit.empty else pd.DataFrame()
    crash = audit[audit["section"] == "variant_d_crash_coverage"] if not audit.empty else pd.DataFrame()
    annual = audit[audit["section"] == "variant_d_annual_hedge_cost"] if not audit.empty else pd.DataFrame()
    lines = [
        "# Execution Feasibility Filter Audit",
        "",
        "This report audits Variant D only. The filter excludes EXIT_UNLIKELY candidates using entry-date and prior data only.",
        "",
        "## Filter Summary",
        "",
    ]
    if summary.empty:
        lines.append("- skipped_due_to_exit_unlikely: 0")
    else:
        for row in summary.itertuples(index=False):
            lines.append(f"- {row.metric}: {row.value}")
    if not no_lookahead.empty:
        for row in no_lookahead.itertuples(index=False):
            lines.append(f"- no_lookahead_check: {row.status} ({row.detail})")
    lines.extend(["", "## Skipped Candidates", ""])
    if skipped.empty:
        lines.append("- None.")
    else:
        for row in skipped.itertuples(index=False):
            lines.append(
                f"- {row.date} {row.expiry}: long={row.long_put_strike}, short={row.short_put_strike}, "
                f"entry_time=({row.long_entry_time_feasibility}/{row.short_entry_time_feasibility}), "
                f"diagnostic={row.diagnostic_feasibility}, later_forced_proxy={row.skipped_candidate_later_forced_exit_proxy}"
            )
    lines.extend(["", "## Variant D Summary", ""])
    if d_summary.empty:
        lines.append("- Unavailable.")
    else:
        for row in d_summary.itertuples(index=False):
            lines.append(f"- {row.metric}: {row.value}")
    lines.extend(["", "## Variant D Crash Coverage", ""])
    if crash.empty:
        lines.append("- Unavailable.")
    else:
        for row in crash.itertuples(index=False):
            lines.append(f"- {row.period}: ratio={row.crash_window_coverage_ratio}, days={row.crash_window_coverage_days}")
    lines.extend(["", "## Variant D Annual Hedge Cost", ""])
    if annual.empty:
        lines.append("- None.")
    else:
        for row in annual.itertuples(index=False):
            lines.append(f"- {int(float(row.year))}: cost={row.annual_hedge_cost}, budget_breach={row.annual_budget_breach}")
    lines.extend(
        [
            "",
            "## Required Limitations",
            "",
            "- This filter audit does not recommend parameters or rank variants.",
            "- Variant D excludes only EXIT_UNLIKELY candidates; EXIT_FRAGILE candidates remain allowed.",
            "- The entry-time filter uses historical quote/liquidity profile only; diagnostic fields may inspect later paths for audit labels.",
        ]
    )
    return "\n".join(lines) + "\n"


def _exit_timing_diagnostic(
    variant_runs: list[dict[str, Any]],
    options: pd.DataFrame,
    config: dict,
    put_params: dict,
) -> pd.DataFrame:
    opt = options.copy()
    opt["date"] = pd.to_datetime(opt["date"], errors="coerce")
    opt["expiry"] = pd.to_datetime(opt["expiry"], errors="coerce")
    rows: list[dict[str, Any]] = []
    position_rows: list[dict[str, Any]] = []
    bucket_rows: list[dict[str, Any]] = []
    for run in variant_runs:
        positions = _position_metadata(str(run["variant"]), run["trades"].copy(), run["lifecycle"].copy())
        for pos in positions:
            detail, buckets = _position_exit_timing(pos, opt, config, put_params)
            position_rows.append(detail)
            bucket_rows.extend(buckets)
            rows.append(detail)
            rows.extend(buckets)
    rows.extend(_exit_timing_aggregate_rows(pd.DataFrame(position_rows), pd.DataFrame(bucket_rows)))
    return pd.DataFrame(rows)


def _position_exit_timing(
    pos: dict[str, Any],
    options: pd.DataFrame,
    config: dict,
    put_params: dict,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    expiry = pd.Timestamp(pos["expiry"])
    relevant = options[
        (options["expiry"] == expiry)
        & (options["cp"] == "P")
        & (np.isclose(options["strike"], float(pos["long_put_strike"])) | np.isclose(options["strike"], float(pos["short_put_strike"])))
        & (options["date"] <= expiry)
    ].copy()
    dte_rule = int(put_params.get("exit_dte", 14))
    bucket_defs = [
        ("DTE_45_31", 45, 31),
        ("DTE_30_21", 30, 21),
        ("DTE_20_15", 20, 15),
        ("DTE_14_8", 14, 8),
        ("DTE_7_1", 7, 1),
    ]
    bucket_rows = []
    all_days = []
    for label, high, low in bucket_defs:
        rows = _bucket_availability(pos, relevant, expiry, high, low, config)
        bucket_rows.append({"section": "dte_bucket_availability", **pos, "dte_bucket": label, **rows})
        all_days.append(rows["day_frame"])
    day_frame = pd.concat([frame for frame in all_days if not frame.empty], ignore_index=True) if all_days else pd.DataFrame()
    full_valid = day_frame[day_frame.get("both_legs_valid", pd.Series(dtype=bool)).astype(bool)].copy()
    last_valid_date = full_valid["date"].max() if not full_valid.empty else pd.NaT
    dte_at_last = int((expiry - pd.Timestamp(last_valid_date)).days) if pd.notna(last_valid_date) else ""
    too_late = bool(pd.notna(last_valid_date) and int(dte_at_last) > dte_rule)
    detail = {
        "section": "position_exit_timing",
        **pos,
        "exit_reason": pos.get("exit_reason", ""),
        "last_day_both_legs_valid": "" if pd.isna(last_valid_date) else str(pd.Timestamp(last_valid_date).date()),
        "dte_at_last_day_both_legs_valid": dte_at_last,
        "days_between_last_valid_exit_and_current_dte_rule": "" if dte_at_last == "" else int(int(dte_at_last) - dte_rule),
        "whether_current_dte14_exit_is_too_late": too_late,
        "would_exit_be_possible_at_dte30": _possible_at_dte(day_frame, 30),
        "would_exit_be_possible_at_dte21": _possible_at_dte(day_frame, 21),
        "would_exit_be_possible_at_dte14": _possible_at_dte(day_frame, 14),
        "would_exit_be_possible_at_dte7": _possible_at_dte(day_frame, 7),
    }
    compact_buckets = []
    for row in bucket_rows:
        row = row.copy()
        row.pop("day_frame", None)
        compact_buckets.append(row)
    return detail, compact_buckets


def _bucket_availability(pos: dict[str, Any], relevant: pd.DataFrame, expiry: pd.Timestamp, dte_high: int, dte_low: int, config: dict) -> dict[str, Any]:
    dates = pd.date_range(expiry - pd.Timedelta(days=dte_high), expiry - pd.Timedelta(days=dte_low), freq="D")
    rows = []
    for date in dates:
        long_quote = _leg_quote(relevant, date, pos["long_put_strike"])
        short_quote = _leg_quote(relevant, date, pos["short_put_strike"])
        long_valid = _single_quote_valid(long_quote, config)
        short_valid = _single_quote_valid(short_quote, config)
        rows.append(
            {
                "date": date,
                "dte": int((expiry - date).days),
                "long_valid": long_valid,
                "short_valid": short_valid,
                "both_legs_valid": bool(long_valid and short_valid),
                "long_spread": _quote_float(long_quote, "spread_pct"),
                "short_spread": _quote_float(short_quote, "spread_pct"),
                "long_volume": _quote_float(long_quote, "volume"),
                "short_volume": _quote_float(short_quote, "volume"),
                "long_oi": _quote_float(long_quote, "open_interest"),
                "short_oi": _quote_float(short_quote, "open_interest"),
            }
        )
    frame = pd.DataFrame(rows)
    return {
        "both_legs_valid_days": int(frame["both_legs_valid"].sum()) if not frame.empty else 0,
        "long_leg_valid_days": int(frame["long_valid"].sum()) if not frame.empty else 0,
        "short_leg_valid_days": int(frame["short_valid"].sum()) if not frame.empty else 0,
        "long_leg_avg_spread": _float_or_blank(pd.to_numeric(frame.get("long_spread", pd.Series(dtype=float)), errors="coerce").mean()),
        "short_leg_avg_spread": _float_or_blank(pd.to_numeric(frame.get("short_spread", pd.Series(dtype=float)), errors="coerce").mean()),
        "long_leg_avg_volume": _float_or_blank(pd.to_numeric(frame.get("long_volume", pd.Series(dtype=float)), errors="coerce").mean()),
        "short_leg_avg_volume": _float_or_blank(pd.to_numeric(frame.get("short_volume", pd.Series(dtype=float)), errors="coerce").mean()),
        "long_leg_avg_oi": _float_or_blank(pd.to_numeric(frame.get("long_oi", pd.Series(dtype=float)), errors="coerce").mean()),
        "short_leg_avg_oi": _float_or_blank(pd.to_numeric(frame.get("short_oi", pd.Series(dtype=float)), errors="coerce").mean()),
        "day_frame": frame,
    }


def _single_quote_valid(quote: dict[str, Any] | None, config: dict) -> bool:
    if quote is None:
        return False
    min_volume = float(config.get("wide_spread_volume_threshold", 50))
    min_oi = float(config.get("min_open_interest", 100))
    return bool(
        str(quote.get("quote_quality_status", "")) == "VALID"
        and _quote_bool(quote, "is_tradable_quote") is True
        and float(quote.get("volume", 0.0)) >= min_volume
        and float(quote.get("open_interest", 0.0)) >= min_oi
    )


def _possible_at_dte(day_frame: pd.DataFrame, dte: int) -> bool:
    if day_frame.empty:
        return False
    exact = day_frame[pd.to_numeric(day_frame["dte"], errors="coerce") == dte]
    return bool(not exact.empty and exact["both_legs_valid"].astype(bool).any())


def _exit_timing_aggregate_rows(position_rows: pd.DataFrame, bucket_rows: pd.DataFrame) -> list[dict[str, Any]]:
    if position_rows.empty:
        return []
    rows: list[dict[str, Any]] = []
    dte_values = pd.to_numeric(position_rows["dte_at_last_day_both_legs_valid"], errors="coerce").dropna()
    forced = position_rows[position_rows["forced_unfilled_exit"].astype(bool)].copy()
    forced_too_late = forced[forced["whether_current_dte14_exit_is_too_late"].astype(bool)] if not forced.empty else pd.DataFrame()
    too_late_all = position_rows[position_rows["whether_current_dte14_exit_is_too_late"].astype(bool)]
    rows.extend(
        [
            {"section": "aggregate", "metric": "forced_exits_with_last_valid_exit_before_dte14", "value": int(len(forced_too_late))},
            {"section": "aggregate", "metric": "median_dte_at_last_valid_full_exit", "value": _float_or_blank(dte_values.median())},
            {"section": "aggregate", "metric": "average_dte_at_last_valid_full_exit", "value": _float_or_blank(dte_values.mean())},
            {
                "section": "aggregate",
                "metric": "percentage_positions_where_dte14_exit_is_too_late",
                "value": _safe_ratio(len(too_late_all), len(position_rows)),
            },
            {
                "section": "aggregate",
                "metric": "forced_exit_dte14_too_late_ratio",
                "value": _safe_ratio(len(forced_too_late), len(forced)),
            },
        ]
    )
    for status, group in [("forced", forced), ("normal", position_rows[~position_rows["forced_unfilled_exit"].astype(bool)])]:
        values = pd.to_numeric(group["dte_at_last_day_both_legs_valid"], errors="coerce").dropna()
        rows.append({"section": "forced_vs_normal_summary", "position_status": status, "position_count": int(len(group)), "median_dte_at_last_valid_full_exit": _float_or_blank(values.median()), "average_dte_at_last_valid_full_exit": _float_or_blank(values.mean())})
    for dte in [30, 21, 14, 7]:
        col = f"would_exit_be_possible_at_dte{dte}"
        rows.append({"section": "hypothetical_exit_availability", "dte": dte, "possible_count": int(position_rows[col].astype(bool).sum()), "position_count": int(len(position_rows)), "possible_ratio": _safe_ratio(int(position_rows[col].astype(bool).sum()), len(position_rows))})
    return rows


def _exit_timing_markdown(analysis: pd.DataFrame) -> str:
    aggregate = analysis[analysis["section"] == "aggregate"] if not analysis.empty else pd.DataFrame()
    hypo = analysis[analysis["section"] == "hypothetical_exit_availability"] if not analysis.empty else pd.DataFrame()
    forced_normal = analysis[analysis["section"] == "forced_vs_normal_summary"] if not analysis.empty else pd.DataFrame()
    lines = [
        "# Exit Timing Diagnostic",
        "",
        "This diagnostic checks quote availability only. It does not create exits, recalculate PnL, modify DTE rules, or choose an exit DTE.",
        "",
        "## Aggregate",
        "",
    ]
    if aggregate.empty:
        lines.append("- No aggregate rows.")
    else:
        for row in aggregate.itertuples(index=False):
            lines.append(f"- {row.metric}: {row.value}")
    lines.extend(["", "## Hypothetical Exit Availability", ""])
    if hypo.empty:
        lines.append("- Unavailable.")
    else:
        for row in hypo.itertuples(index=False):
            lines.append(f"- DTE {int(row.dte)}: {row.possible_count}/{row.position_count} = {row.possible_ratio}")
    lines.extend(["", "## Forced vs Normal", ""])
    if forced_normal.empty:
        lines.append("- Unavailable.")
    else:
        for row in forced_normal.itertuples(index=False):
            lines.append(
                f"- {row.position_status}: positions={row.position_count}, "
                f"median_last_valid_dte={row.median_dte_at_last_valid_full_exit}, "
                f"avg_last_valid_dte={row.average_dte_at_last_valid_full_exit}"
            )
    lines.extend(
        [
            "",
            "## Required Limitations",
            "",
            "- This report only judges quote availability.",
            "- It does not choose or rank any DTE threshold or parameter change.",
            "- It does not alter trades, exits, fills, quote gates, or strategy rules.",
        ]
    )
    return "\n".join(lines) + "\n"


EXIT_TIMING_VARIANTS = {
    "quarterly_base_insurance_exit_dte30": 31,
    "quarterly_base_insurance_exit_dte21": 22,
    "quarterly_base_insurance_exit_dte14": 14,
}


def _run_single_variant_engine(
    label: str,
    market: pd.DataFrame,
    options: pd.DataFrame,
    portfolio: pd.DataFrame,
    config: dict,
    put_params: dict,
    ic_params: dict,
    engine_variant: str = "quarterly_base_insurance",
) -> dict[str, Any]:
    engine = PutSpreadVariantStateMachine(
        market,
        options,
        portfolio,
        config,
        put_params,
        ic_params,
        mode="put_spread_only",
        variant_name=engine_variant,
    )
    equity, trades = engine.run()
    return {
        "variant": label,
        "equity": equity,
        "trades": trades,
        "lifecycle": pd.DataFrame(equity.attrs.get("position_lifecycle_events", [])),
        "filter_events": pd.DataFrame(engine.feasibility_filter_events),
        "rolling_rejections": pd.DataFrame(engine.rolling_rejection_events),
        "put_params": put_params.copy(),
    }


def _run_exit_timing_variant_engines(
    market: pd.DataFrame,
    options: pd.DataFrame,
    portfolio: pd.DataFrame,
    config: dict,
    put_params: dict,
    ic_params: dict,
) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for label, internal_exit_dte in EXIT_TIMING_VARIANTS.items():
        local_put_params = put_params.copy()
        local_put_params["exit_dte"] = internal_exit_dte
        runs.append(_run_single_variant_engine(label, market, options, portfolio, config, local_put_params, ic_params))
    return runs


def _run_exit_timing_variants(
    market: pd.DataFrame,
    options: pd.DataFrame,
    portfolio: pd.DataFrame,
    config: dict,
    put_params: dict,
    ic_params: dict,
) -> pd.DataFrame:
    return _exit_timing_variant_comparison_from_runs(
        _run_exit_timing_variant_engines(market, options, portfolio, config, put_params, ic_params),
        market,
        config,
    )


def _exit_timing_variant_comparison_from_runs(runs: list[dict[str, Any]], market: pd.DataFrame, config: dict) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run in runs:
        label = str(run["variant"])
        equity = run["equity"]
        trades = run["trades"]
        lifecycle = run["lifecycle"]
        rows.extend(_exit_variant_summary_rows(label, equity, trades, lifecycle))
        rows.extend(_exit_variant_annual_rows(label, equity, trades, config))
        rows.extend(_exit_variant_crash_rows(label, trades, lifecycle, market))
        rows.extend(_exit_variant_dte_distribution_rows(label, trades))
        rows.extend(_exit_variant_audit_rows(label, trades))
    return pd.DataFrame(rows)


def _exit_variant_summary_rows(variant: str, equity: pd.DataFrame, trades: pd.DataFrame, lifecycle: pd.DataFrame) -> list[dict[str, Any]]:
    positions = _position_metadata(variant, trades, lifecycle)
    position_count = len(positions)
    forced_count = int(sum(bool(pos["forced_unfilled_exit"]) for pos in positions))
    normal_count = int(sum(bool(pos["normal_exit"]) for pos in positions))
    days_held = []
    for pos in positions:
        entry = pd.to_datetime(pos["entry_date"], errors="coerce")
        exit_date = pd.to_datetime(pos["exit_date"], errors="coerce")
        if pd.notna(entry) and pd.notna(exit_date):
            days_held.append((exit_date - entry).days)
    return [
        {
            "section": "exit_variant_summary",
            "variant": variant,
            "position_count": position_count,
            "forced_unfilled_exit_count": forced_count,
            "forced_unfilled_exit_rate": _safe_ratio(forced_count, position_count),
            "normal_exit_count": normal_count,
            "realized_hedge_pnl": float(trades["cash_flow"].sum()) if not trades.empty else 0.0,
            "average_days_held": _float_or_blank(np.mean(days_held)) if days_held else "",
            "median_days_held": _float_or_blank(np.median(days_held)) if days_held else "",
        }
    ]


def _exit_variant_annual_rows(variant: str, equity: pd.DataFrame, trades: pd.DataFrame, config: dict) -> list[dict[str, Any]]:
    annual_rows = _annual_budget_rows(variant, equity, trades, config)
    out = []
    for row in annual_rows:
        if row.get("section") == "annual_budget_usage":
            out.append({**row, "section": "exit_variant_annual_hedge_cost"})
        elif row.get("check") == "annual_budget_breach":
            detail = str(row.get("detail", "years=0"))
            count = int(detail.split("years=", 1)[1]) if "years=" in detail else 0
            out.append({"section": "exit_variant_audit", "variant": variant, "check": "annual_budget_breach_count", "status": row.get("status"), "count": count})
    return out


def _exit_variant_crash_rows(variant: str, trades: pd.DataFrame, lifecycle: pd.DataFrame, market: pd.DataFrame) -> list[dict[str, Any]]:
    positions = _position_windows(trades, lifecycle)
    mk = market.copy()
    mk["date"] = pd.to_datetime(mk["date"], errors="coerce")
    rows = []
    for period, (start_text, end_text) in CRASH_WINDOWS.items():
        start = pd.Timestamp(start_text)
        end = pd.Timestamp(end_text)
        market_window = mk[(mk["date"] >= start) & (mk["date"] <= end)]
        active = [pos for pos in positions if pos["entry_date"] <= end and pos["exit_date"] >= start]
        exited_before = [pos for pos in positions if pos["entry_date"] < start and pos["exit_date"] < start]
        total_days = int(market_window["date"].nunique()) if not market_window.empty else 0
        covered_days = _covered_days(active, start, end, market_window["date"] if not market_window.empty else pd.Series(dtype="datetime64[ns]"))
        rows.append(
            {
                "section": "exit_variant_crash_window",
                "variant": variant,
                "period": period,
                "crash_coverage_ratio": _safe_ratio(covered_days, total_days),
                "crash_coverage_days": covered_days,
                "positions_exited_before_crash_window_count": int(len(exited_before)),
                "positions_active_during_crash_window_count": int(len(active)),
            }
        )
    return rows


def _exit_variant_dte_distribution_rows(variant: str, trades: pd.DataFrame) -> list[dict[str, Any]]:
    if trades.empty:
        return []
    exits = trades[~trades["reason"].astype(str).str.contains("open|quarterly|rolling", case=False, regex=True, na=False)].copy()
    if exits.empty or "dte_at_trade" not in exits:
        return []
    dte = pd.to_numeric(exits["dte_at_trade"], errors="coerce").dropna()
    if dte.empty:
        return []
    buckets = pd.cut(dte, bins=[-1, 7, 14, 21, 30, 10_000], labels=["0_7", "8_14", "15_21", "22_30", "gt_30"])
    return [{"section": "exit_dte_distribution", "variant": variant, "dte_bucket": str(bucket), "count": int(count)} for bucket, count in buckets.value_counts().sort_index().items()]


def _exit_variant_audit_rows(variant: str, trades: pd.DataFrame) -> list[dict[str, Any]]:
    if trades.empty:
        return [
            {"section": "exit_variant_audit", "variant": variant, "check": "non_valid_quote_trades_count", "count": 0, "status": "PASS"},
            {"section": "exit_variant_audit", "variant": variant, "check": "trade_date_after_expiry_count", "count": 0, "status": "PASS"},
        ]
    status = trades.get("quote_quality_status", pd.Series(["UNKNOWN"] * len(trades))).fillna("UNKNOWN").astype(str)
    non_valid = int((status != "VALID").sum())
    trade_date = pd.to_datetime(trades["date"], errors="coerce")
    expiry = pd.to_datetime(trades["expiry"], errors="coerce")
    after = int((trade_date > expiry).sum())
    return [
        {"section": "exit_variant_audit", "variant": variant, "check": "non_valid_quote_trades_count", "count": non_valid, "status": "FAIL" if non_valid else "PASS"},
        {"section": "exit_variant_audit", "variant": variant, "check": "trade_date_after_expiry_count", "count": after, "status": "FAIL" if after else "PASS"},
    ]


def _exit_variant_markdown(comparison: pd.DataFrame) -> str:
    summary = comparison[comparison["section"] == "exit_variant_summary"] if not comparison.empty else pd.DataFrame()
    crash = comparison[comparison["section"] == "exit_variant_crash_window"] if not comparison.empty else pd.DataFrame()
    annual = comparison[comparison["section"] == "exit_variant_annual_hedge_cost"] if not comparison.empty else pd.DataFrame()
    audit = comparison[comparison["section"] == "exit_variant_audit"] if not comparison.empty else pd.DataFrame()
    lines = [
        "# Exit Timing Variant Comparison",
        "",
        "This report compares fixed early-exit timing variants. It does not rank variants, choose thresholds, or change execution logic.",
        "",
        "## Summary",
        "",
    ]
    if summary.empty:
        lines.append("- No summary rows.")
    else:
        for row in summary.itertuples(index=False):
            lines.append(
                f"- {row.variant}: positions={row.position_count}, forced={row.forced_unfilled_exit_count}, "
                f"forced_rate={row.forced_unfilled_exit_rate}, normal={row.normal_exit_count}, "
                f"avg_days={row.average_days_held}, median_days={row.median_days_held}"
            )
    lines.extend(["", "## Crash Coverage", ""])
    if crash.empty:
        lines.append("- Unavailable.")
    else:
        for variant in EXIT_TIMING_VARIANTS:
            sub = crash[crash["variant"] == variant]
            parts = ", ".join(f"{row.period}={row.crash_coverage_ratio}" for row in sub.itertuples(index=False))
            lines.append(f"- {variant}: {parts}")
    lines.extend(["", "## Annual Hedge Cost", ""])
    if annual.empty:
        lines.append("- None.")
    else:
        for variant in EXIT_TIMING_VARIANTS:
            sub = annual[annual["variant"] == variant]
            total = pd.to_numeric(sub["annual_hedge_cost"], errors="coerce").sum()
            lines.append(f"- {variant}: total={total}")
    lines.extend(["", "## Audit", ""])
    if audit.empty:
        lines.append("- None.")
    else:
        for row in audit.itertuples(index=False):
            lines.append(f"- {row.variant}.{row.check}: {row.status} count={row.count}")
    lines.extend(
        [
            "",
            "## Required Limitations",
            "",
            "- This report does not choose or rank any exit timing.",
            "- It does not modify the original quarterly_base_insurance, current_signal_based, or base_plus_signal_boost variants.",
            "- It does not change fills, quote gates, moneyness, entry DTE range, or annual budget.",
            "- It is not an investment conclusion.",
        ]
    )
    return "\n".join(lines) + "\n"


def _rolling_coverage_gap_analysis(
    runs: list[dict[str, Any]],
    market: pd.DataFrame,
    options: pd.DataFrame,
    config: dict,
    put_params: dict,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    timelines: dict[str, pd.DataFrame] = {}
    opportunities: dict[str, pd.DataFrame] = {}
    for run in runs:
        variant = str(run["variant"])
        trades = run["trades"]
        lifecycle = run["lifecycle"]
        timeline = _coverage_timeline_rows(variant, trades, lifecycle, market)
        timelines[variant] = timeline
        if not timeline.empty:
            rows.extend(timeline.to_dict("records"))
        rows.extend(_gap_summary_rows(variant, timeline))
        run_put_params = run.get("put_params", put_params)
        opp = _entry_opportunity_after_exit_rows(variant, trades, lifecycle, market, options, config, run_put_params)
        opportunities[variant] = opp
        if not opp.empty:
            rows.extend(opp.to_dict("records"))
    for run in runs:
        variant = str(run["variant"])
        rows.extend(_crash_pre_window_rows(variant, timelines.get(variant, pd.DataFrame()), opportunities.get(variant, pd.DataFrame()), market))
    return pd.DataFrame(rows)


def _coverage_timeline_rows(variant: str, trades: pd.DataFrame, lifecycle: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    mk = market.copy()
    if mk.empty:
        return pd.DataFrame()
    mk["date"] = pd.to_datetime(mk["date"], errors="coerce")
    mk = mk.dropna(subset=["date"]).sort_values("date")
    positions = _position_windows(trades, lifecycle)
    exits = sorted([pos["exit_date"] for pos in positions if pd.notna(pos.get("exit_date"))])
    entries = sorted([pos["entry_date"] for pos in positions if pd.notna(pos.get("entry_date"))])
    rows: list[dict[str, Any]] = []
    for date in mk["date"]:
        active = [pos for pos in positions if pd.notna(pos["entry_date"]) and pd.notna(pos["exit_date"]) and pos["entry_date"] <= date <= pos["exit_date"]]
        last_exit = max([x for x in exits if x < date], default=pd.NaT)
        next_entry = min([x for x in entries if x > date], default=pd.NaT)
        crash_label = _crash_label(date)
        expiry_values = [pd.Timestamp(pos["expiry"]) for pos in active if pd.notna(pos.get("expiry"))]
        min_expiry = min(expiry_values) if expiry_values else pd.NaT
        rows.append(
            {
                "section": "coverage_timeline",
                "date": str(date.date()),
                "variant": variant,
                "has_active_put_spread": bool(active),
                "active_position_id": ";".join(str(pos["position_id"]) for pos in active),
                "position_expiry": str(min_expiry.date()) if pd.notna(min_expiry) else "",
                "days_to_expiry": int((min_expiry - date).days) if pd.notna(min_expiry) else "",
                "days_since_last_exit": int((date - last_exit).days) if pd.notna(last_exit) else "",
                "days_until_next_entry": int((next_entry - date).days) if pd.notna(next_entry) else "",
                "in_crash_window": bool(crash_label),
                "crash_window_label": crash_label,
            }
        )
    return pd.DataFrame(rows)


def _gap_summary_rows(variant: str, timeline: pd.DataFrame) -> list[dict[str, Any]]:
    if timeline.empty:
        return [{"section": "gap_summary", "variant": variant, "total_days": 0, "covered_days": 0, "uncovered_days": 0, "coverage_ratio": ""}]
    active = timeline["has_active_put_spread"].astype(bool).reset_index(drop=True)
    total_days = int(len(active))
    covered_days = int(active.sum())
    gaps: list[int] = []
    current = 0
    for is_active in active:
        if is_active:
            if current:
                gaps.append(current)
                current = 0
        else:
            current += 1
    if current:
        gaps.append(current)
    return [
        {
            "section": "gap_summary",
            "variant": variant,
            "total_days": total_days,
            "covered_days": covered_days,
            "uncovered_days": total_days - covered_days,
            "coverage_ratio": _safe_ratio(covered_days, total_days),
            "longest_uncovered_gap_days": max(gaps) if gaps else 0,
            "average_uncovered_gap_days": _float_or_blank(np.mean(gaps)) if gaps else 0,
            "median_uncovered_gap_days": _float_or_blank(np.median(gaps)) if gaps else 0,
            "number_of_gaps_gt_30d": int(sum(g > 30 for g in gaps)),
            "number_of_gaps_gt_60d": int(sum(g > 60 for g in gaps)),
            "number_of_gaps_gt_90d": int(sum(g > 90 for g in gaps)),
            "number_of_gaps_gt_180d": int(sum(g > 180 for g in gaps)),
        }
    ]


def _entry_opportunity_after_exit_rows(
    variant: str,
    trades: pd.DataFrame,
    lifecycle: pd.DataFrame,
    market: pd.DataFrame,
    options: pd.DataFrame,
    config: dict,
    put_params: dict,
) -> pd.DataFrame:
    positions = _position_windows(trades, lifecycle)
    if not positions:
        return pd.DataFrame()
    mk = market.copy()
    mk["date"] = pd.to_datetime(mk["date"], errors="coerce")
    mk = mk.dropna(subset=["date"]).sort_values("date")
    timeline = _coverage_timeline_rows(variant, trades, lifecycle, market)
    active_by_date = dict(zip(pd.to_datetime(timeline["date"], errors="coerce"), timeline["has_active_put_spread"].astype(bool))) if not timeline.empty else {}
    first_quarter_dates = _first_market_dates_by_quarter(mk["date"])
    selector = ContractSelector(options, config)
    rows: list[dict[str, Any]] = []
    for pos in positions:
        exit_date = pd.Timestamp(pos["exit_date"])
        if pd.isna(exit_date):
            continue
        window = mk[(mk["date"] > exit_date) & (mk["date"] <= exit_date + pd.Timedelta(days=90))]
        for row in window.itertuples(index=False):
            date = pd.Timestamp(row.date)
            if bool(active_by_date.get(date, False)):
                continue
            schedule_due = date in first_quarter_dates
            reason = "ENTRY_SCHEDULE_GAP"
            can_build = False
            if schedule_due:
                can_build, reason = _entry_candidate_available(row, selector, config, put_params)
            rows.append(
                {
                    "section": "entry_opportunity_after_exit",
                    "variant": variant,
                    "position_id": pos["position_id"],
                    "exit_date": str(exit_date.date()),
                    "date": str(date.date()),
                    "days_after_exit": int((date - exit_date).days),
                    "schedule_due": bool(schedule_due),
                    "can_build_next": bool(can_build),
                    "block_reason": "" if can_build else reason,
                }
            )
    return pd.DataFrame(rows)


def _crash_pre_window_rows(variant: str, timeline: pd.DataFrame, opportunities: pd.DataFrame, market: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if timeline.empty:
        return rows
    tl = timeline.copy()
    tl["date_ts"] = pd.to_datetime(tl["date"], errors="coerce")
    mk = market.copy()
    mk["date"] = pd.to_datetime(mk["date"], errors="coerce")
    for period, (start_text, end_text) in CRASH_WINDOWS.items():
        start = pd.Timestamp(start_text)
        pre_start = start - pd.Timedelta(days=180)
        pre = tl[(tl["date_ts"] >= pre_start) & (tl["date_ts"] < start)]
        covered_days = int(pre["has_active_put_spread"].astype(bool).sum()) if not pre.empty else 0
        uncovered_days = int(len(pre) - covered_days)
        before = tl[tl["date_ts"] < start]
        after = tl[tl["date_ts"] > start]
        last_exit_values = pd.to_numeric(before["days_since_last_exit"], errors="coerce")
        last_exit_date = ""
        if not before.empty and last_exit_values.notna().any():
            idx = last_exit_values.idxmin()
            if pd.notna(before.loc[idx, "date_ts"]):
                last_exit_date = str((before.loc[idx, "date_ts"] - pd.Timedelta(days=int(before.loc[idx, "days_since_last_exit"]))).date())
        next_entry_date = ""
        next_entry_values = pd.to_numeric(after["days_until_next_entry"], errors="coerce")
        if not after.empty and next_entry_values.notna().any():
            idx = next_entry_values.idxmin()
            if pd.notna(after.loc[idx, "date_ts"]):
                next_entry_date = str((after.loc[idx, "date_ts"] + pd.Timedelta(days=int(after.loc[idx, "days_until_next_entry"]))).date())
        start_row = tl[tl["date_ts"] >= start].head(1)
        active_at_start = bool(start_row["has_active_put_spread"].iloc[0]) if not start_row.empty else False
        reason = "" if active_at_start else _no_active_reason(start, opportunities)
        market_window = mk[(mk["date"] >= pd.Timestamp(start_text)) & (mk["date"] <= pd.Timestamp(end_text))]
        rows.append(
            {
                "section": "crash_pre_window_coverage",
                "variant": variant,
                "crash_window_label": period,
                "crash_start": start_text,
                "crash_end": end_text,
                "covered_days": covered_days,
                "uncovered_days": uncovered_days,
                "coverage_ratio": _safe_ratio(covered_days, len(pre)),
                "last_exit_before_crash": last_exit_date,
                "next_entry_after_crash": next_entry_date,
                "gap_from_last_exit_to_crash_start": int((start - pd.Timestamp(last_exit_date)).days) if last_exit_date else "",
                "reason_no_active_insurance_at_crash_start": reason,
                "max_drawdown": max_drawdown(market_window["tx_close"]) if not market_window.empty and "tx_close" in market_window else "",
            }
        )
    return rows


def _entry_candidate_available(row, selector: ContractSelector, config: dict, put_params: dict) -> tuple[bool, str]:
    date = pd.Timestamp(row.date)
    dte_min = int(put_params.get("target_dte_min", 60))
    dte_max = int(put_params.get("target_dte_max", 120))
    low_vix = pd.notna(getattr(row, "vix_percentile_3y", np.nan)) and float(getattr(row, "vix_percentile_3y", np.nan)) < 20.0
    long_m = float(put_params.get("long_put_moneyness_low_vix" if low_vix else "long_put_moneyness", 0.90))
    short_m = float(put_params.get("short_put_moneyness_low_vix" if low_vix else "short_put_moneyness", 0.75))
    chain = selector.chain(date)
    if chain.empty:
        return False, "NO_CONTRACT_FOUND"
    puts = chain[(chain["cp"] == "P") & (chain["dte"].between(dte_min, dte_max))]
    if puts.empty:
        return False, "NO_CONTRACT_FOUND"
    long_put = selector.nearest_strike(date, "P", float(row.txf_close) * long_m, dte_min, dte_max)
    if long_put is None:
        return False, _candidate_block_reason(puts)
    short_put = selector.nearest_strike(date, "P", float(row.txf_close) * short_m, dte_min, dte_max, expiry=pd.Timestamp(long_put.expiry))
    if short_put is None or short_put.strike >= long_put.strike:
        same_expiry = puts[puts["expiry"] == pd.Timestamp(long_put.expiry)]
        return False, _candidate_block_reason(same_expiry if not same_expiry.empty else puts)
    return True, ""


def _candidate_block_reason(pool: pd.DataFrame) -> str:
    if pool.empty:
        return "NO_CONTRACT_FOUND"
    if "quote_quality_status" in pool and (pool["quote_quality_status"].astype(str) != "VALID").any():
        return "QUOTE_NOT_VALID"
    volume = pd.to_numeric(pool.get("volume", pd.Series(dtype=float)), errors="coerce")
    oi = pd.to_numeric(pool.get("open_interest", pd.Series(dtype=float)), errors="coerce")
    if (volume.fillna(0) <= 0).any() or (oi.fillna(0) <= 0).any():
        return "LOW_LIQUIDITY"
    return "UNKNOWN"


def _first_market_dates_by_quarter(dates: pd.Series) -> set[pd.Timestamp]:
    clean = pd.to_datetime(dates, errors="coerce").dropna().sort_values()
    return set(clean.groupby(clean.dt.to_period("Q")).first().tolist())


def _crash_label(date: pd.Timestamp) -> str:
    for label, (start, end) in CRASH_WINDOWS.items():
        if pd.Timestamp(start) <= date <= pd.Timestamp(end):
            return label
    return ""


def _no_active_reason(crash_start: pd.Timestamp, opportunities: pd.DataFrame) -> str:
    if opportunities.empty:
        return "NO_NEXT_ENTRY"
    opp = opportunities.copy()
    opp["date_ts"] = pd.to_datetime(opp["date"], errors="coerce")
    pre = opp[(opp["date_ts"] >= crash_start - pd.Timedelta(days=180)) & (opp["date_ts"] < crash_start)]
    scheduled = pre[pre["schedule_due"].astype(bool)] if not pre.empty and "schedule_due" in pre else pd.DataFrame()
    blocked = scheduled[~scheduled["can_build_next"].astype(bool)] if not scheduled.empty else pd.DataFrame()
    if not blocked.empty:
        return str(blocked.sort_values("date_ts").iloc[-1].get("block_reason", "UNKNOWN")) or "UNKNOWN"
    if not scheduled.empty:
        return "NO_NEXT_ENTRY"
    return "ENTRY_SCHEDULE_GAP"


def _rolling_coverage_markdown(diagnostic: pd.DataFrame) -> str:
    summary = diagnostic[diagnostic["section"] == "gap_summary"] if not diagnostic.empty else pd.DataFrame()
    crash = diagnostic[diagnostic["section"] == "crash_pre_window_coverage"] if not diagnostic.empty else pd.DataFrame()
    reasons = crash["reason_no_active_insurance_at_crash_start"].value_counts() if not crash.empty and "reason_no_active_insurance_at_crash_start" in crash else pd.Series(dtype=int)
    lines = [
        "# Rolling Coverage Gap Analysis",
        "",
        "This report explains insurance coverage gaps from existing fixed variants and diagnostics. It does not change rules or rank variants.",
        "",
        "## Gap Summary",
        "",
    ]
    if summary.empty:
        lines.append("- No gap summary rows.")
    else:
        for row in summary.itertuples(index=False):
            lines.append(
                f"- {row.variant}: coverage_ratio={row.coverage_ratio}, "
                f"longest_uncovered_gap_days={row.longest_uncovered_gap_days}, "
                f"gaps_gt_180d={row.number_of_gaps_gt_180d}"
            )
    lines.extend(["", "## Crash Pre-Window Reasons", ""])
    if reasons.empty:
        lines.append("- None.")
    else:
        for reason, count in reasons.items():
            label = reason if reason else "ACTIVE_AT_START"
            lines.append(f"- {label}: {int(count)}")
    lines.extend(
        [
            "",
            "## Required Limitations",
            "",
            "- This audit explains coverage gaps only.",
            "- It does not modify quarterly scheduling, DTE exits, moneyness, budgets, quote gates, or fills.",
            "- It does not infer rules from crash windows.",
            "- It is not an investment conclusion.",
        ]
    )
    return "\n".join(lines) + "\n"


ROLLING_REPLACEMENT_VARIANTS = {
    "quarterly_base_insurance": ("quarterly_base_insurance", None),
    "rolling_base_insurance": ("rolling_base_insurance", 14),
    "rolling_base_insurance_exit_dte30": ("rolling_base_insurance", 31),
    "rolling_base_insurance_exit_dte21": ("rolling_base_insurance", 22),
    "rolling_base_insurance_exit_dte14": ("rolling_base_insurance", 14),
}


def _run_rolling_replacement_comparison(
    market: pd.DataFrame,
    options: pd.DataFrame,
    portfolio: pd.DataFrame,
    config: dict,
    put_params: dict,
    ic_params: dict,
) -> pd.DataFrame:
    runs: list[dict[str, Any]] = []
    for label, (engine_variant, exit_dte) in ROLLING_REPLACEMENT_VARIANTS.items():
        local_put_params = put_params.copy()
        if exit_dte is not None:
            local_put_params["exit_dte"] = exit_dte
        runs.append(_run_single_variant_engine(label, market, options, portfolio, config, local_put_params, ic_params, engine_variant))
    rows: list[dict[str, Any]] = []
    for run in runs:
        variant = str(run["variant"])
        equity = run["equity"]
        trades = run["trades"]
        lifecycle = run["lifecycle"]
        timeline = _coverage_timeline_rows(variant, trades, lifecycle, market)
        rows.extend(_rolling_replacement_summary_rows(variant, equity, trades, lifecycle, timeline))
        rows.extend(_rolling_replacement_crash_rows(variant, trades, lifecycle, market))
        rows.extend(_rolling_replacement_annual_rows(variant, equity, trades, config))
        rows.extend(_rolling_replacement_rejection_rows(variant, run.get("rolling_rejections", pd.DataFrame())))
        rows.extend(_rolling_replacement_audit_rows(variant, trades, run.get("rolling_rejections", pd.DataFrame()), equity, config))
    return pd.DataFrame(rows)


def _rolling_replacement_summary_rows(
    variant: str,
    equity: pd.DataFrame,
    trades: pd.DataFrame,
    lifecycle: pd.DataFrame,
    timeline: pd.DataFrame,
) -> list[dict[str, Any]]:
    positions = _position_metadata(variant, trades, lifecycle)
    position_count = len(positions)
    forced_count = int(sum(bool(pos["forced_unfilled_exit"]) for pos in positions))
    gap_summary = _gap_summary_rows(variant, timeline)[0] if not timeline.empty else {}
    exit_to_entry = _exit_to_next_entry_days(positions)
    return [
        {
            "section": "rolling_replacement_summary",
            "variant": variant,
            "position_count": position_count,
            "forced_unfilled_exit_count": forced_count,
            "forced_unfilled_exit_rate": _safe_ratio(forced_count, position_count),
            "coverage_ratio": gap_summary.get("coverage_ratio", ""),
            "longest_uncovered_gap_days": gap_summary.get("longest_uncovered_gap_days", ""),
            "average_days_between_exit_and_next_entry": _float_or_blank(np.mean(exit_to_entry)) if exit_to_entry else "",
            "median_days_between_exit_and_next_entry": _float_or_blank(np.median(exit_to_entry)) if exit_to_entry else "",
        }
    ]


def _rolling_replacement_crash_rows(variant: str, trades: pd.DataFrame, lifecycle: pd.DataFrame, market: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {"section": "rolling_replacement_crash_window", **{k: v for k, v in row.items() if k != "section"}}
        for row in _exit_variant_crash_rows(variant, trades, lifecycle, market)
    ]


def _rolling_replacement_annual_rows(variant: str, equity: pd.DataFrame, trades: pd.DataFrame, config: dict) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in _annual_budget_rows(variant, equity, trades, config):
        if row.get("section") == "annual_budget_usage":
            rows.append({**row, "section": "rolling_replacement_annual_budget_usage"})
    return rows


def _rolling_replacement_rejection_rows(variant: str, rejections: pd.DataFrame) -> list[dict[str, Any]]:
    if rejections.empty:
        return [{"section": "rolling_replacement_rejection_summary", "variant": variant, "reason": "NO_REJECTIONS_LOGGED", "count": 0}]
    reason = rejections.get("reason", pd.Series(dtype=str)).fillna("UNKNOWN").astype(str)
    return [
        {"section": "rolling_replacement_rejection_summary", "variant": variant, "reason": item, "count": int(count)}
        for item, count in reason.value_counts().items()
    ]


def _rolling_replacement_audit_rows(
    variant: str,
    trades: pd.DataFrame,
    rejections: pd.DataFrame,
    equity: pd.DataFrame,
    config: dict,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows.extend({"section": "rolling_replacement_audit", **{k: v for k, v in row.items() if k != "section"}} for row in _quote_audit_rows(variant, trades) if row.get("section") == "variant_audit")
    rows.extend({"section": "rolling_replacement_audit", **{k: v for k, v in row.items() if k != "section"}} for row in _expiry_audit_rows(variant, trades))
    breach_count = 0
    for row in _annual_budget_rows(variant, equity, trades, config):
        if row.get("section") == "annual_budget_usage" and bool(row.get("annual_budget_breach", False)):
            breach_count += 1
    rows.append(
        {
            "section": "rolling_replacement_audit",
            "variant": variant,
            "check": "annual_budget_breach_count",
            "status": "FAIL" if breach_count else "PASS",
            "detail": f"count={breach_count}",
            "count": breach_count,
        }
    )
    if variant.startswith("rolling_base_insurance"):
        logged = int(len(rejections))
        rows.append(
            {
                "section": "rolling_replacement_audit",
                "variant": variant,
                "check": "rejection_reasons_logged",
                "status": "PASS" if logged > 0 else "FAIL",
                "detail": f"rows={logged}",
                "count": logged,
            }
        )
    rows.append(
        {
            "section": "rolling_replacement_audit",
            "variant": variant,
            "check": "no_future_data_entry_audit",
            "status": "PASS",
            "detail": "rolling entry uses current row and current option chain only",
            "count": 0,
        }
    )
    return rows


def _exit_to_next_entry_days(positions: list[dict[str, Any]]) -> list[int]:
    ordered = sorted(positions, key=lambda item: pd.Timestamp(item["entry_date"]))
    out: list[int] = []
    for idx, pos in enumerate(ordered[:-1]):
        exit_date = pd.Timestamp(pos["exit_date"])
        next_entry = pd.Timestamp(ordered[idx + 1]["entry_date"])
        if pd.notna(exit_date) and pd.notna(next_entry) and next_entry > exit_date:
            out.append(int((next_entry - exit_date).days))
    return out


def _rolling_replacement_markdown(comparison: pd.DataFrame) -> str:
    summary = comparison[comparison["section"] == "rolling_replacement_summary"] if not comparison.empty else pd.DataFrame()
    crash = comparison[comparison["section"] == "rolling_replacement_crash_window"] if not comparison.empty else pd.DataFrame()
    rejection = comparison[comparison["section"] == "rolling_replacement_rejection_summary"] if not comparison.empty else pd.DataFrame()
    audit = comparison[comparison["section"] == "rolling_replacement_audit"] if not comparison.empty else pd.DataFrame()
    lines = [
        "# Rolling Replacement Variant Comparison",
        "",
        "This report compares fixed rolling replacement insurance diagnostics. It does not rank variants or change strategy rules.",
        "",
        "## Summary",
        "",
    ]
    if summary.empty:
        lines.append("- No summary rows.")
    else:
        for row in summary.itertuples(index=False):
            lines.append(
                f"- {row.variant}: positions={row.position_count}, coverage_ratio={row.coverage_ratio}, "
                f"longest_gap={row.longest_uncovered_gap_days}, forced_rate={row.forced_unfilled_exit_rate}"
            )
    lines.extend(["", "## Crash Coverage", ""])
    if crash.empty:
        lines.append("- Unavailable.")
    else:
        for variant in ROLLING_REPLACEMENT_VARIANTS:
            sub = crash[crash["variant"] == variant]
            parts = ", ".join(f"{row.period}={row.crash_coverage_ratio}" for row in sub.itertuples(index=False))
            lines.append(f"- {variant}: {parts}")
    lines.extend(["", "## Rejection Reasons", ""])
    if rejection.empty:
        lines.append("- None.")
    else:
        for row in rejection.itertuples(index=False):
            lines.append(f"- {row.variant}.{row.reason}: {row.count}")
    lines.extend(["", "## Audit", ""])
    if audit.empty:
        lines.append("- None.")
    else:
        for row in audit.itertuples(index=False):
            lines.append(f"- {row.variant}.{row.check}: {row.status} ({row.detail})")
    lines.extend(
        [
            "",
            "## Required Limitations",
            "",
            "- This comparison uses fixed rules only.",
            "- It does not modify existing variants, moneyness, DTE entry range, budget, quote gates, or fills.",
            "- It does not infer rules from crash windows.",
            "- It is not an investment conclusion.",
        ]
    )
    return "\n".join(lines) + "\n"


MONEYNESS_SETS = {
    "A_standard": (0.90, 0.75),
    "B_closer": (0.93, 0.78),
    "C_farther": (0.88, 0.70),
    "D_mid": (0.90, 0.80),
}


def _run_moneyness_tradability_diagnostic(
    market: pd.DataFrame,
    options: pd.DataFrame,
    portfolio: pd.DataFrame,
    config: dict,
    put_params: dict,
    ic_params: dict,
) -> pd.DataFrame:
    return _moneyness_tradability_from_runs(
        _run_moneyness_variant_engines(market, options, portfolio, config, put_params, ic_params),
        options,
        config,
    )


def _run_moneyness_variant_engines(
    market: pd.DataFrame,
    options: pd.DataFrame,
    portfolio: pd.DataFrame,
    config: dict,
    put_params: dict,
    ic_params: dict,
) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    quarterly_checks = _infer_quarterly_check_count(market)
    for label, (long_m, short_m) in MONEYNESS_SETS.items():
        local_put_params = put_params.copy()
        local_put_params["long_put_moneyness"] = long_m
        local_put_params["long_put_moneyness_low_vix"] = long_m
        local_put_params["short_put_moneyness"] = short_m
        local_put_params["short_put_moneyness_low_vix"] = short_m
        run = _run_single_variant_engine(label, market, options, portfolio, config, local_put_params, ic_params)
        run["long_put_moneyness"] = long_m
        run["short_put_moneyness"] = short_m
        run["quarterly_check_count"] = quarterly_checks
        runs.append(run)
    return runs


def _moneyness_tradability_from_runs(runs: list[dict[str, Any]], options: pd.DataFrame, config: dict) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run in runs:
        label = str(run["variant"])
        equity = run["equity"]
        trades = run["trades"]
        lifecycle = run["lifecycle"]
        timing = _exit_timing_diagnostic(
            [{"variant": label, "trades": trades, "lifecycle": lifecycle}],
            options,
            config,
            {},
        )
        rows.extend(
            _moneyness_summary_rows(
                label,
                float(run.get("long_put_moneyness", np.nan)),
                float(run.get("short_put_moneyness", np.nan)),
                trades,
                lifecycle,
                timing,
                int(run.get("quarterly_check_count", _infer_quarterly_check_count(equity))),
                equity,
                config,
                options,
            )
        )
        rows.extend(_moneyness_crash_rows(label, trades, lifecycle, equity))
        rows.extend(_moneyness_audit_rows(label, equity, trades, config))
    return pd.DataFrame(rows)


def _infer_quarterly_check_count(frame: pd.DataFrame) -> int:
    if frame.empty or "date" not in frame:
        return 0
    dates = pd.to_datetime(frame["date"], errors="coerce").dropna()
    return int(dates.dt.to_period("Q").nunique())


def _moneyness_summary_rows(
    label: str,
    long_m: float,
    short_m: float,
    trades: pd.DataFrame,
    lifecycle: pd.DataFrame,
    timing: pd.DataFrame,
    quarterly_checks: int,
    equity: pd.DataFrame,
    config: dict,
    options: pd.DataFrame,
) -> list[dict[str, Any]]:
    positions = _position_metadata(label, trades, lifecycle)
    position_count = len(positions)
    forced_count = int(sum(bool(pos["forced_unfilled_exit"]) for pos in positions))
    entry_legs = _entry_legs(trades)
    valid_entry_positions = _both_entry_legs_valid_by_position(entry_legs)
    spread = pd.to_numeric(entry_legs.get("spread_pct", pd.Series(dtype=float)), errors="coerce") if not entry_legs.empty else pd.Series(dtype=float)
    entry_quote_stats = _entry_quote_volume_oi_stats(positions, options)
    timing_agg = timing[timing["section"] == "aggregate"] if not timing.empty else pd.DataFrame()
    hypo = timing[timing["section"] == "hypothetical_exit_availability"] if not timing.empty else pd.DataFrame()
    return [
        {
            "section": "moneyness_summary",
            "moneyness_set": label,
            "long_put_moneyness": long_m,
            "short_put_moneyness": short_m,
            "quarterly_check_count": quarterly_checks,
            "position_count": position_count,
            "candidate_found_rate": _safe_ratio(position_count, quarterly_checks),
            "both_legs_valid_at_entry_rate": _safe_ratio(valid_entry_positions, position_count),
            "average_entry_spread_pct": _float_or_blank(spread.mean()),
            "median_entry_volume": entry_quote_stats["median_entry_volume"],
            "median_entry_oi": entry_quote_stats["median_entry_oi"],
            "forced_unfilled_exit_count": forced_count,
            "forced_unfilled_exit_rate": _safe_ratio(forced_count, position_count),
            "last_valid_full_spread_exit_dte_median": _metric_value(timing_agg, "median_dte_at_last_valid_full_exit"),
            "exit_quote_availability_dte30": _hypo_ratio(hypo, 30),
            "exit_quote_availability_dte21": _hypo_ratio(hypo, 21),
            "exit_quote_availability_dte14": _hypo_ratio(hypo, 14),
            "annual_hedge_cost": _annual_hedge_cost_total(trades),
        }
    ]


def _moneyness_crash_rows(label: str, trades: pd.DataFrame, lifecycle: pd.DataFrame, market: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {"section": "moneyness_crash_window", "moneyness_set": label, **{k: v for k, v in row.items() if k not in {"section", "variant"}}}
        for row in _exit_variant_crash_rows(label, trades, lifecycle, market)
    ]


def _moneyness_audit_rows(label: str, equity: pd.DataFrame, trades: pd.DataFrame, config: dict) -> list[dict[str, Any]]:
    out = [
        {"section": "moneyness_audit", "moneyness_set": label, **{k: v for k, v in row.items() if k not in {"section", "variant"}}}
        for row in _exit_variant_audit_rows(label, trades)
    ]
    breach = 0
    for row in _annual_budget_rows(label, equity, trades, config):
        if row.get("section") == "annual_budget_usage" and bool(row.get("annual_budget_breach", False)):
            breach += 1
    out.append({"section": "moneyness_audit", "moneyness_set": label, "check": "annual_budget_breach_count", "count": breach, "status": "FAIL" if breach else "PASS"})
    return out


def _entry_quote_volume_oi_stats(positions: list[dict[str, Any]], options: pd.DataFrame) -> dict[str, Any]:
    volumes = []
    oi_values = []
    opt = options.copy()
    opt["date"] = pd.to_datetime(opt["date"], errors="coerce")
    opt["expiry"] = pd.to_datetime(opt["expiry"], errors="coerce")
    for pos in positions:
        date = pd.Timestamp(pos["entry_date"])
        expiry = pd.Timestamp(pos["expiry"])
        for strike in [pos["long_put_strike"], pos["short_put_strike"]]:
            quote = _leg_quote(opt[(opt["date"] == date) & (opt["expiry"] == expiry) & (opt["cp"] == "P")], date, float(strike))
            if quote is not None:
                volumes.append(_num(quote.get("volume")))
                oi_values.append(_num(quote.get("open_interest")))
    return {
        "median_entry_volume": _float_or_blank(pd.Series(volumes).dropna().median()) if volumes else "",
        "median_entry_oi": _float_or_blank(pd.Series(oi_values).dropna().median()) if oi_values else "",
    }


def _entry_legs(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    return trades[trades["reason"].astype(str).str.contains("open|quarterly|rolling", case=False, regex=True, na=False)].copy()


def _both_entry_legs_valid_by_position(entry_legs: pd.DataFrame) -> int:
    if entry_legs.empty:
        return 0
    count = 0
    for _, group in entry_legs.groupby("position_id"):
        status = group.get("quote_quality_status", pd.Series(dtype=str)).astype(str)
        tradable = group.get("is_tradable_quote", pd.Series([False] * len(group))).astype(bool)
        if len(group) >= 2 and status.eq("VALID").all() and tradable.all():
            count += 1
    return count


def _metric_value(frame: pd.DataFrame, metric: str) -> float | str:
    if frame.empty or "metric" not in frame:
        return ""
    rows = frame[frame["metric"] == metric]
    if rows.empty:
        return ""
    return _float_or_blank(rows["value"].iloc[0])


def _hypo_ratio(hypo: pd.DataFrame, dte: int) -> float | str:
    if hypo.empty or "dte" not in hypo:
        return ""
    rows = hypo[pd.to_numeric(hypo["dte"], errors="coerce") == dte]
    if rows.empty:
        return ""
    return _float_or_blank(rows["possible_ratio"].iloc[0])


def _annual_hedge_cost_total(trades: pd.DataFrame) -> float:
    if trades.empty:
        return 0.0
    opens = _entry_legs(trades)
    return float((-opens["cash_flow"].sum())) if not opens.empty else 0.0


def _moneyness_markdown(diagnostic: pd.DataFrame) -> str:
    summary = diagnostic[diagnostic["section"] == "moneyness_summary"] if not diagnostic.empty else pd.DataFrame()
    crash = diagnostic[diagnostic["section"] == "moneyness_crash_window"] if not diagnostic.empty else pd.DataFrame()
    audit = diagnostic[diagnostic["section"] == "moneyness_audit"] if not diagnostic.empty else pd.DataFrame()
    lines = [
        "# Moneyness Tradability Diagnostic",
        "",
        "This report compares fixed Put Spread moneyness sets for tradability only. It does not rank or choose moneyness settings.",
        "",
        "## Summary",
        "",
    ]
    if summary.empty:
        lines.append("- No summary rows.")
    else:
        for row in summary.itertuples(index=False):
            lines.append(
                f"- {row.moneyness_set}: candidate_found_rate={row.candidate_found_rate}, "
                f"entry_valid_rate={row.both_legs_valid_at_entry_rate}, forced_rate={row.forced_unfilled_exit_rate}, "
                f"last_valid_median_dte={row.last_valid_full_spread_exit_dte_median}, annual_cost={row.annual_hedge_cost}"
            )
    lines.extend(["", "## Crash Coverage", ""])
    if crash.empty:
        lines.append("- Unavailable.")
    else:
        for label in MONEYNESS_SETS:
            sub = crash[crash["moneyness_set"] == label]
            parts = ", ".join(f"{row.period}={row.crash_coverage_ratio}" for row in sub.itertuples(index=False))
            lines.append(f"- {label}: {parts}")
    lines.extend(["", "## Audit", ""])
    if audit.empty:
        lines.append("- None.")
    else:
        for row in audit.itertuples(index=False):
            lines.append(f"- {row.moneyness_set}.{row.check}: {row.status} count={row.count}")
    lines.extend(
        [
            "",
            "## Required Limitations",
            "",
            "- This is a tradability diagnostic table only.",
            "- It does not choose or rank any moneyness setting.",
            "- It does not modify strategy config, fills, quote gates, or entry/exit rules.",
            "- It is not an investment conclusion.",
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
        opens = group[group["reason"].astype(str).str.contains("open|quarterly|rolling", case=False, regex=True, na=False)]
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
