"""Diagnostics-only risk-scaled put-spread coverage simulation.

This module simulates the institutional behavior of:
HedgeNeedScore -> target_hedge_coverage -> hedge_gap -> execution attempt.
It does not change HedgeNeedScore weights, target mapping, or execution fills.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .contracts import Leg, Position, StrategyState
from .data_loader import load_macro_factors, load_market, load_options, load_portfolio
from .execution import fill_price, is_liquid, is_stress_day, trade_contract
from .hedge_need import apply_risk_indicators_to_market, hedge_need_score_frame, load_risk_indicators
from .indicators import add_market_indicators
from .macro_regime import add_macro_regime_indicators, macro_restrictions
from .portfolio import make_state
from .put_spread_variants import CRASH_WINDOWS
from .strategies import BlackSwanStateMachine
from .utils import year_key


DEFAULT_HEDGE_GAP_THRESHOLD = 0.03
DEFAULT_MAX_CONTRACTS_PER_DAY = 10
DEFAULT_MAX_TOTAL_HEDGE_COVERAGE = 0.80
BUDGET_POLICIES = [
    "current_policy",
    "score_gated_budget_release",
    "reserve_high_risk_budget",
    "base_then_risk_budget",
]


def calculate_hedge_gap(target_hedge_coverage: float, current_hedge_coverage: float) -> float:
    """Return target minus current hedge coverage."""

    return float(target_hedge_coverage) - float(current_hedge_coverage)


def budget_release_fraction(policy: str, hedge_need_score: float) -> float:
    """Return annual budget fraction released by a fixed diagnostics policy."""

    score = float(np.clip(hedge_need_score, 0.0, 100.0)) if pd.notna(hedge_need_score) else 0.0
    if policy == "current_policy":
        return 1.0
    if policy == "score_gated_budget_release":
        if score < 25.0:
            return 0.10
        if score < 50.0:
            return 0.40
        if score < 75.0:
            return 0.75
        return 1.0
    if policy == "reserve_high_risk_budget":
        return 0.60 if score < 50.0 else 1.0
    if policy == "base_then_risk_budget":
        if score < 25.0:
            return 0.10
        if score < 50.0:
            return 0.50
        return 1.0
    raise ValueError(f"Unknown budget policy: {policy}")


def _put_spread_entry_cost_per_contract(long_price: float, short_price: float, config: dict) -> float:
    """Return expected cash cost for opening one long/short put spread."""

    point_value = float(config["txo_point_value"])
    commission = float(config["commission_per_contract_per_side"])
    tax_rate = float(config["option_tax_rate_on_premium"])
    long_gross = float(long_price) * point_value
    short_gross = float(short_price) * point_value
    return (long_gross - short_gross) + (2.0 * commission) + ((long_gross + short_gross) * tax_rate)


class RiskScaledHedgeStateMachine(BlackSwanStateMachine):
    """Put-spread simulation driven only by target hedge coverage gap."""

    def __init__(self, *args, budget_policy: str = "current_policy", **kwargs):
        super().__init__(*args, **kwargs)
        if budget_policy not in BUDGET_POLICIES:
            raise ValueError(f"Unknown budget policy: {budget_policy}")
        self.budget_policy = budget_policy
        self.position_counter = itertools.count(1)
        self.coverage_rows: list[dict[str, Any]] = []
        self.rejection_events: list[dict[str, Any]] = []

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        prev_stock = None
        prev_option_value = None
        for row in self.market.itertuples(index=False):
            date = pd.Timestamp(row.date)
            expiry_warning = self._expire_past_due_positions(date)
            stock_equity = self._stock_equity(date)
            self._refresh_open_legs(date)
            option_value = self._option_liquidation_value(row)
            required_margin = self._required_margin()
            warning = self._risk_update(row, required_margin, option_value, stock_equity)
            if expiry_warning:
                warning = ";".join(filter(None, [warning, expiry_warning]))
            self._check_exits(row)
            self._check_put_entry(row, stock_equity)
            option_value = self._option_liquidation_value(row)
            required_margin = self._required_margin()
            state = make_state(
                str(date.date()),
                self.cash,
                stock_equity,
                option_value,
                required_margin,
                self.state,
                prev_stock,
                prev_option_value,
                warning,
                str(getattr(row, "macro_state", "NORMAL")),
                float(getattr(row, "SupplyStressIndex", 0.0)),
                float(getattr(row, "MacroDemandFragilityIndex", 0.0)),
                float(getattr(row, "CombinedRiskScore", 0.0)),
            )
            self.states.append(asdict(state))
            prev_stock = stock_equity
            prev_option_value = option_value
        equity = pd.DataFrame(self.states)
        equity.attrs["position_lifecycle_events"] = self.lifecycle_events
        return equity, pd.DataFrame([asdict(t) for t in self.trades])

    def _stock_equity(self, date: pd.Timestamp) -> float:
        stock_row = self.portfolio[self.portfolio["date"] == date]
        if not stock_row.empty:
            return float(stock_row.iloc[0]["stock_equity"])
        return float(self.config["initial_stock_equity"])

    def _check_put_entry(self, row, stock_equity: float) -> None:
        date = pd.Timestamp(row.date)
        beta_exposure = stock_equity * float(self.config.get("portfolio_beta", 1.0))
        current_coverage = current_put_spread_coverage(
            self._open_positions("put_spread"),
            beta_exposure,
            float(self.config["txo_point_value"]),
        )
        target_coverage = float(getattr(row, "target_hedge_coverage", 0.0))
        hedge_gap = calculate_hedge_gap(target_coverage, current_coverage)
        threshold = float(self.config.get("risk_scaled_hedge_gap_threshold", DEFAULT_HEDGE_GAP_THRESHOLD))
        budget_cap, budget_used, budget_remaining = self._annual_budget(row, stock_equity)
        record = self._base_coverage_record(row, stock_equity, beta_exposure, target_coverage, current_coverage, hedge_gap, budget_cap, budget_used, budget_remaining)
        record["attempted_entry_today"] = False
        record["entry_success"] = False
        record["rejection_reason"] = ""

        if hedge_gap <= threshold:
            record["rejection_reason"] = "HEDGE_GAP_TOO_SMALL"
            self.coverage_rows.append(record)
            return
        if current_coverage >= float(self.config.get("risk_scaled_max_total_hedge_coverage", DEFAULT_MAX_TOTAL_HEDGE_COVERAGE)):
            record["attempted_entry_today"] = True
            record["rejection_reason"] = "MAX_COVERAGE_REACHED"
            self._log_rejection(record)
            self.coverage_rows.append(record)
            return
        if budget_remaining <= 0:
            record["attempted_entry_today"] = True
            record["rejection_reason"] = "BUDGET_EXCEEDED"
            self._log_rejection(record)
            self.coverage_rows.append(record)
            return

        record["attempted_entry_today"] = True
        candidate = self._select_put_spread_candidate(row)
        record.update(candidate["detail"])
        if not candidate["ok"]:
            record["rejection_reason"] = candidate["reason"]
            self._log_rejection(record)
            self.coverage_rows.append(record)
            return
        long_put = candidate["long_put"]
        short_put = candidate["short_put"]
        stress = is_stress_day(row, self.config)
        long_px = fill_price(long_put, "BUY", self.config, stress)
        short_px = fill_price(short_put, "SELL", self.config, stress)
        per_spread_cost = _put_spread_entry_cost_per_contract(long_px, short_px, self.config)
        if per_spread_cost <= 0:
            record["rejection_reason"] = "UNKNOWN"
            self._log_rejection(record)
            self.coverage_rows.append(record)
            return
        width_protection = (long_put.strike - short_put.strike) * float(self.config["txo_point_value"])
        qty = self._quantity_for_gap(hedge_gap, beta_exposure, current_coverage, width_protection, per_spread_cost, budget_remaining)
        if qty <= 0:
            record["rejection_reason"] = "BUDGET_EXCEEDED"
            self._log_rejection(record)
            self.coverage_rows.append(record)
            return

        opened = self._open_put_spread(row, long_put, short_put, qty, stress)
        record["entry_success"] = opened
        if opened:
            spend = -(self.trades[-2].cash_flow + self.trades[-1].cash_flow)
            self.annual_hedge_spend[year_key(row.date)] = budget_used + max(0.0, spend)
            _, updated_used, updated_remaining = self._annual_budget(row, stock_equity)
            after_coverage = current_put_spread_coverage(
                self._open_positions("put_spread"),
                beta_exposure,
                float(self.config["txo_point_value"]),
            )
            record["annual_budget_used"] = updated_used
            record["annual_budget_remaining"] = updated_remaining
            record["current_hedge_coverage_after_entry"] = after_coverage
            record["hedge_gap_after_entry"] = calculate_hedge_gap(target_coverage, after_coverage)
            record["contracts_added"] = qty
            record["position_id"] = self.positions[-1].id
        else:
            record["rejection_reason"] = "UNKNOWN"
            self._log_rejection(record)
        self.coverage_rows.append(record)

    def _annual_budget(self, row, stock_equity: float) -> tuple[float, float, float]:
        macro_state = str(getattr(row, "macro_state", "NORMAL"))
        restriction = macro_restrictions(macro_state)
        cap = float(self.config["max_annual_hedge_budget_pct"]) * float(restriction["hedge_budget_multiplier"]) * stock_equity
        used = self.annual_hedge_spend.get(year_key(row.date), 0.0)
        release_cap = cap * budget_release_fraction(self.budget_policy, float(getattr(row, "HedgeNeedScore", 0.0)))
        return cap, used, max(0.0, min(cap, release_cap) - used)

    def _base_coverage_record(
        self,
        row,
        stock_equity: float,
        beta_exposure: float,
        target_coverage: float,
        current_coverage: float,
        hedge_gap: float,
        budget_cap: float,
        budget_used: float,
        budget_remaining: float,
    ) -> dict[str, Any]:
        return {
            "date": str(pd.Timestamp(row.date).date()),
            "HedgeNeedScore": float(getattr(row, "HedgeNeedScore", np.nan)),
            "target_hedge_coverage": target_coverage,
            "current_hedge_coverage": current_coverage,
            "current_hedge_coverage_after_entry": current_coverage,
            "hedge_gap": hedge_gap,
            "hedge_gap_after_entry": hedge_gap,
            "stock_equity": stock_equity,
            "portfolio_beta_exposure": beta_exposure,
            "active_put_spread_count": len(self._open_positions("put_spread")),
            "annual_budget_limit": budget_cap,
            "annual_budget_used": budget_used,
            "annual_budget_remaining": budget_remaining,
            "budget_policy": self.budget_policy,
            "score_confidence": str(getattr(row, "score_confidence", "")),
            "volatility_source_type": str(getattr(row, "volatility_source_type", "")),
            "contracts_added": 0,
            "position_id": "",
        }

    def _select_put_spread_candidate(self, row) -> dict[str, Any]:
        date = pd.Timestamp(row.date)
        restriction = macro_restrictions(str(getattr(row, "macro_state", "NORMAL")))
        dte_min = int(self.put_params["target_dte_min"])
        dte_max = int(self.put_params["target_dte_max"]) + int(restriction["extend_put_dte"])
        low_vix = pd.notna(getattr(row, "vix_percentile_3y", np.nan)) and float(getattr(row, "vix_percentile_3y", np.nan)) < 20.0
        long_m = float(self.put_params["long_put_moneyness_low_vix"] if low_vix else self.put_params["long_put_moneyness"])
        short_m = float(self.put_params["short_put_moneyness_low_vix"] if low_vix else self.put_params["short_put_moneyness"])
        detail = {
            "target_dte_min": dte_min,
            "target_dte_max": dte_max,
            "long_target_moneyness": long_m,
            "short_target_moneyness": short_m,
        }
        chain = self.selector.chain(date)
        if chain.empty:
            return {"ok": False, "reason": "NO_CONTRACT_FOUND", "detail": detail}
        puts = chain[(chain["cp"] == "P") & (chain["dte"].between(dte_min, dte_max))]
        if puts.empty:
            return {"ok": False, "reason": "NO_CONTRACT_FOUND", "detail": detail}
        long_put = self.selector.nearest_strike(date, "P", float(row.txf_close) * long_m, dte_min, dte_max)
        if long_put is None:
            return {"ok": False, "reason": _candidate_block_reason(puts), "detail": detail}
        short_put = self.selector.nearest_strike(date, "P", float(row.txf_close) * short_m, dte_min, dte_max, expiry=pd.Timestamp(long_put.expiry))
        if short_put is None or short_put.strike >= long_put.strike:
            same_expiry = puts[puts["expiry"] == pd.Timestamp(long_put.expiry)]
            return {"ok": False, "reason": _candidate_block_reason(same_expiry if not same_expiry.empty else puts), "detail": detail}
        detail.update(
            {
                "long_put_strike": long_put.strike,
                "short_put_strike": short_put.strike,
                "expiry": long_put.expiry,
                "long_quote_quality_status": long_put.quote_quality_status,
                "short_quote_quality_status": short_put.quote_quality_status,
                "long_is_tradable_quote": long_put.is_tradable_quote,
                "short_is_tradable_quote": short_put.is_tradable_quote,
            }
        )
        return {"ok": True, "reason": "", "detail": detail, "long_put": long_put, "short_put": short_put}

    def _quantity_for_gap(
        self,
        hedge_gap: float,
        beta_exposure: float,
        current_coverage: float,
        width_protection: float,
        per_spread_cost: float,
        budget_remaining: float,
    ) -> int:
        if width_protection <= 0 or per_spread_cost <= 0:
            return 0
        max_total_coverage = float(self.config.get("risk_scaled_max_total_hedge_coverage", DEFAULT_MAX_TOTAL_HEDGE_COVERAGE))
        max_contracts = int(self.config.get("risk_scaled_max_contracts_per_day", DEFAULT_MAX_CONTRACTS_PER_DAY))
        max_by_gap = math.ceil(max(0.0, hedge_gap) * beta_exposure / width_protection)
        max_by_coverage = math.floor(max(0.0, max_total_coverage - current_coverage) * beta_exposure / width_protection)
        max_by_budget = math.floor(max(0.0, budget_remaining) / per_spread_cost)
        return max(0, min(max_by_gap, max_by_coverage, max_by_budget, max_contracts))

    def _open_put_spread(self, row, long_put, short_put, qty: int, stress: bool) -> bool:
        if qty <= 0:
            return False
        date = pd.Timestamp(row.date)
        pos_id = f"RSH-{next(self.position_counter)}"
        fill1, tr1 = trade_contract(str(date.date()), pos_id, "put_spread", long_put, "BUY", qty, self.config, "risk_scaled_hedge_open", stress)
        fill2, tr2 = trade_contract(str(date.date()), pos_id, "put_spread", short_put, "SELL", qty, self.config, "risk_scaled_hedge_open", stress)
        self.cash += fill1.cash_flow + fill2.cash_flow
        self.trades.extend([tr1, tr2])
        spend = -(fill1.cash_flow + fill2.cash_flow)
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
                notes={"remaining_entry_cost": spend, "variant": "risk_scaled_hedge_simulation"},
            )
        )
        self.state = StrategyState.HEDGE_ON
        return True

    def _log_rejection(self, record: dict[str, Any]) -> None:
        self.rejection_events.append(
            {
                "date": record["date"],
                "rejection_reason": record.get("rejection_reason", "UNKNOWN") or "UNKNOWN",
                "HedgeNeedScore": record.get("HedgeNeedScore", np.nan),
                "hedge_gap": record.get("hedge_gap", np.nan),
                "annual_budget_remaining": record.get("annual_budget_remaining", np.nan),
            }
        )


def current_put_spread_coverage(positions: list[Position], portfolio_beta_exposure: float, point_value: float) -> float:
    """Return active put-spread max protection divided by beta exposure."""

    if portfolio_beta_exposure <= 0:
        return 0.0
    protection = 0.0
    for pos in positions:
        if pos.closed or pos.strategy != "put_spread":
            continue
        long_legs = [leg for leg in pos.legs if leg.quantity > 0 and leg.cp == "P"]
        short_legs = [leg for leg in pos.legs if leg.quantity < 0 and leg.cp == "P"]
        if not long_legs or not short_legs:
            continue
        width = max(0.0, long_legs[0].strike - short_legs[0].strike)
        protection += width * point_value * abs(pos.quantity)
    return float(protection / portfolio_beta_exposure)


def run_risk_scaled_hedge_simulation(data_dir: Path, report_dir: Path, config: dict, put_params: dict, ic_params: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run diagnostics-only risk-scaled coverage simulation and write reports."""

    report_dir.mkdir(parents=True, exist_ok=True)
    market = load_market(data_dir)
    if market.empty:
        out = pd.DataFrame([{"section": "error", "status": "FAIL", "detail": "market.csv unavailable"}])
        out.to_csv(report_dir / "risk_scaled_hedge_simulation.csv", index=False)
        (report_dir / "risk_scaled_hedge_simulation.md").write_text(_markdown(out, pd.DataFrame(), pd.DataFrame()), encoding="utf-8")
        return out, pd.DataFrame(), pd.DataFrame()

    market = apply_risk_indicators_to_market(market, load_risk_indicators(data_dir))
    market = add_market_indicators(market)
    market = add_macro_regime_indicators(market, load_macro_factors(data_dir))
    score = hedge_need_score_frame(market)
    score_cols = [
        "date",
        "HedgeNeedScore",
        "target_hedge_coverage",
        "vix_proxy_in_use",
        "iv_proxy_source",
        "volatility_source_type",
        "macro_data_quality_flag",
    ]
    market = market.merge(score[score_cols], on="date", how="left", suffixes=("", "_score"))
    attribution_path = report_dir / "hedge_need_score_attribution.csv"
    if attribution_path.exists():
        attribution = pd.read_csv(attribution_path, usecols=lambda col: col in {"section", "date", "score_confidence"})
        daily_conf = attribution[attribution["section"] == "daily_attribution"][["date", "score_confidence"]].copy()
        daily_conf["date"] = pd.to_datetime(daily_conf["date"], errors="coerce")
        market = market.merge(daily_conf, on="date", how="left")
    else:
        market["score_confidence"] = ""
    load_config = config.copy()
    load_config["runtime_mode"] = "put_spread_only"
    options = load_options(data_dir, market, load_config)
    portfolio = load_portfolio(data_dir, market, config)

    engine = RiskScaledHedgeStateMachine(market, options, portfolio, config, put_params, ic_params, mode="put_spread_only")
    equity, trades = engine.run()
    coverage = pd.DataFrame(engine.coverage_rows)
    lifecycle = pd.DataFrame(equity.attrs.get("position_lifecycle_events", []))
    summary = _summary_rows(coverage, trades, lifecycle, equity, config)
    summary.extend(_crash_coverage_rows(coverage))
    summary.extend(_annual_budget_rows(trades, equity, config))
    summary.extend(_audit_rows(trades, coverage, config))
    out = pd.DataFrame(summary)
    breakdown = risk_scaled_hedge_breakdown(coverage, trades, lifecycle, equity, config)
    budget_audit = hedge_budget_allocation_audit(coverage, trades, equity, config)
    dynamic_budget = dynamic_budget_policy_diagnostic(
        market,
        options,
        portfolio,
        config,
        put_params,
        ic_params,
        current_run={"policy": "current_policy", "equity": equity, "trades": trades, "coverage": coverage, "lifecycle": lifecycle},
    )

    out.to_csv(report_dir / "risk_scaled_hedge_simulation.csv", index=False)
    trades.to_csv(report_dir / "risk_scaled_hedge_trades.csv", index=False)
    coverage.to_csv(report_dir / "risk_scaled_hedge_coverage.csv", index=False)
    breakdown.to_csv(report_dir / "risk_scaled_hedge_breakdown.csv", index=False)
    budget_audit.to_csv(report_dir / "hedge_budget_allocation_audit.csv", index=False)
    dynamic_budget.to_csv(report_dir / "dynamic_budget_policy_diagnostic.csv", index=False)
    (report_dir / "risk_scaled_hedge_simulation.md").write_text(_markdown(out, coverage, trades), encoding="utf-8")
    (report_dir / "risk_scaled_hedge_breakdown.md").write_text(_breakdown_markdown(breakdown), encoding="utf-8")
    (report_dir / "hedge_budget_allocation_audit.md").write_text(_budget_audit_markdown(budget_audit), encoding="utf-8")
    (report_dir / "dynamic_budget_policy_diagnostic.md").write_text(_dynamic_budget_policy_markdown(dynamic_budget), encoding="utf-8")
    return out, trades, coverage


def hedge_budget_allocation_audit(coverage: pd.DataFrame, trades: pd.DataFrame, equity: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Build diagnostics-only audit of annual hedge budget allocation."""

    cov = coverage.copy()
    if cov.empty:
        return pd.DataFrame([{"section": "error", "status": "FAIL", "detail": "coverage unavailable"}])
    cov["date"] = pd.to_datetime(cov["date"], errors="coerce")
    cov["year"] = cov["date"].dt.year
    tr = trades.copy()
    if not tr.empty:
        tr["date"] = pd.to_datetime(tr["date"], errors="coerce")
        tr["expiry"] = pd.to_datetime(tr["expiry"], errors="coerce") if "expiry" in tr else pd.NaT
        tr["year"] = tr["date"].dt.year
    eq = equity.copy()
    if not eq.empty:
        eq["date"] = pd.to_datetime(eq["date"], errors="coerce")
        eq["year"] = eq["date"].dt.year
    daily_cost = _daily_new_hedge_cost(tr)
    out = _budget_timeline_rows(cov, daily_cost)
    out.extend(_budget_score_bucket_rows(cov, tr, daily_cost))
    out.extend(_budget_exhaustion_rows(cov))
    out.extend(_budget_crash_pre_window_rows(cov))
    out.extend(_budget_crash_classification_rows(cov))
    return pd.DataFrame(out)


def dynamic_budget_policy_diagnostic(
    market: pd.DataFrame,
    options: pd.DataFrame,
    portfolio: pd.DataFrame,
    config: dict,
    put_params: dict,
    ic_params: dict,
    current_run: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Compare fixed annual-budget release policies without changing total budget."""

    rows: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    if current_run is not None:
        runs.append(current_run)
    for policy in BUDGET_POLICIES:
        if current_run is not None and policy == current_run.get("policy"):
            continue
        engine = RiskScaledHedgeStateMachine(market, options, portfolio, config, put_params, ic_params, mode="put_spread_only", budget_policy=policy)
        equity, trades = engine.run()
        runs.append(
            {
                "policy": policy,
                "equity": equity,
                "trades": trades,
                "coverage": pd.DataFrame(engine.coverage_rows),
                "lifecycle": pd.DataFrame(equity.attrs.get("position_lifecycle_events", [])),
            }
        )
    for run in runs:
        policy = str(run["policy"])
        cov = run["coverage"].copy()
        trades = run["trades"].copy()
        equity = run["equity"].copy()
        lifecycle = run["lifecycle"].copy()
        if not cov.empty:
            cov["date"] = pd.to_datetime(cov["date"], errors="coerce")
            cov["year"] = cov["date"].dt.year
        if not trades.empty:
            trades["date"] = pd.to_datetime(trades["date"], errors="coerce")
            trades["expiry"] = pd.to_datetime(trades["expiry"], errors="coerce") if "expiry" in trades else pd.NaT
            trades["year"] = trades["date"].dt.year
        if not equity.empty:
            equity["date"] = pd.to_datetime(equity["date"], errors="coerce")
            equity["year"] = equity["date"].dt.year
        rows.extend(_dynamic_policy_summary_rows(policy, cov, trades, equity, lifecycle, config))
        rows.extend(_dynamic_policy_score_bucket_rows(policy, cov, trades))
        rows.extend(_dynamic_policy_crash_rows(policy, cov))
        rows.extend(_dynamic_policy_audit_rows(policy, cov, trades, lifecycle))
    return pd.DataFrame(rows)


def risk_scaled_hedge_breakdown(
    coverage: pd.DataFrame,
    trades: pd.DataFrame,
    lifecycle: pd.DataFrame,
    equity: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """Build diagnostics-only breakdown of risk-scaled simulation behavior."""

    rows: list[dict[str, Any]] = []
    cov = coverage.copy()
    if not cov.empty:
        cov["date"] = pd.to_datetime(cov["date"], errors="coerce")
        cov["year"] = cov["date"].dt.year
    tr = trades.copy()
    if not tr.empty:
        tr["date"] = pd.to_datetime(tr["date"], errors="coerce")
        if "expiry" in tr:
            tr["expiry"] = pd.to_datetime(tr["expiry"], errors="coerce")
        tr["year"] = tr["date"].dt.year
    eq = equity.copy()
    if not eq.empty:
        eq["date"] = pd.to_datetime(eq["date"], errors="coerce")
        eq["year"] = eq["date"].dt.year
    rows.extend(_breakdown_annual_cost(cov, tr, eq, config))
    rows.extend(_breakdown_coverage_gap(cov))
    rows.extend(_breakdown_crash_windows(cov, tr))
    rows.extend(_breakdown_forced_exits(cov, tr, lifecycle))
    rows.extend(_breakdown_rejections(cov))
    return pd.DataFrame(rows)


def _candidate_block_reason(pool: pd.DataFrame) -> str:
    if pool.empty:
        return "NO_CONTRACT_FOUND"
    if "quote_quality_status" in pool and (pool["quote_quality_status"].astype(str) != "VALID").any():
        return "QUOTE_NOT_VALID"
    if "is_tradable_quote" in pool and (~pool["is_tradable_quote"].astype(bool)).any():
        return "QUOTE_NOT_VALID"
    volume = pd.to_numeric(pool.get("volume", pd.Series(dtype=float)), errors="coerce")
    oi = pd.to_numeric(pool.get("open_interest", pd.Series(dtype=float)), errors="coerce")
    if (volume.fillna(0) < 50).any() or (oi.fillna(0) < 100).any():
        return "LOW_LIQUIDITY"
    return "UNKNOWN"


def _dynamic_policy_summary_rows(policy: str, cov: pd.DataFrame, trades: pd.DataFrame, equity: pd.DataFrame, lifecycle: pd.DataFrame, config: dict) -> list[dict[str, Any]]:
    annual_cost = _annual_cost_total(trades)
    forced = int((lifecycle.get("issue", pd.Series(dtype=str)).astype(str) == "forced_unfilled_exit").sum()) if not lifecycle.empty else 0
    budget_block = int(cov["rejection_reason"].astype(str).eq("BUDGET_EXCEEDED").sum()) if not cov.empty else 0
    quote_block = int(cov["rejection_reason"].astype(str).eq("QUOTE_NOT_VALID").sum()) if not cov.empty else 0
    return [
        {
            "section": "policy_summary",
            "policy": policy,
            "annual_hedge_cost": annual_cost,
            "average_hedge_gap": _mean(cov, "hedge_gap_after_entry"),
            "average_current_coverage": _mean(cov, "current_hedge_coverage_after_entry"),
            "days_execution_blocked_by_budget": budget_block,
            "days_execution_blocked_by_quote": quote_block,
            "forced_unfilled_exit_count": forced,
        }
    ]


def _dynamic_policy_score_bucket_rows(policy: str, cov: pd.DataFrame, trades: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    daily_cost = _daily_new_hedge_cost(trades)
    if cov.empty:
        return rows
    data = cov.copy()
    data["date_norm"] = data["date"].dt.normalize()
    data["score_bucket"] = data["HedgeNeedScore"].map(_score_bucket)
    data["new_hedge_cost_today"] = data["date_norm"].map(daily_cost).fillna(0.0)
    entries = _position_entry_buckets(trades, data)
    entry_counts = _entry_count_by_bucket(trades, entries)
    for bucket in ["0-25", "25-50", "50-75", "75-100"]:
        group = data[data["score_bucket"].eq(bucket)]
        rows.append(
            {
                "section": "policy_score_bucket",
                "policy": policy,
                "score_bucket": bucket,
                "hedge_cost_spent": float(group["new_hedge_cost_today"].sum()),
                "number_of_entries": int(entry_counts.get(bucket, 0)),
                "average_hedge_gap": float(pd.to_numeric(group["hedge_gap_after_entry"], errors="coerce").mean()) if not group.empty else np.nan,
            }
        )
    return rows


def _dynamic_policy_crash_rows(policy: str, cov: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if cov.empty:
        return rows
    for label, (start_text, end_text) in CRASH_WINDOWS.items():
        start = pd.Timestamp(start_text)
        end = pd.Timestamp(end_text)
        crash = cov[(cov["date"] >= start) & (cov["date"] <= end)]
        pre = cov[(cov["date"] >= start - pd.Timedelta(days=180)) & (cov["date"] < start)]
        row: dict[str, Any] = {
            "section": "policy_crash_window",
            "policy": policy,
            "crash_window": label,
            "crash_coverage_ratio": float((pd.to_numeric(crash["current_hedge_coverage_after_entry"], errors="coerce").fillna(0) > 0).mean()) if not crash.empty else np.nan,
        }
        for offset in [180, 90, 60, 30]:
            sample_date = start - pd.Timedelta(days=offset)
            prior = cov[cov["date"] <= sample_date].sort_values("date").tail(1)
            if prior.empty:
                row[f"budget_remaining_{offset}d_before_crash"] = ""
                row[f"budget_used_pct_{offset}d_before_crash"] = ""
                continue
            sample = prior.iloc[0]
            used = float(sample.get("annual_budget_used", 0.0))
            limit = float(sample.get("annual_budget_limit", used + float(sample.get("annual_budget_remaining", 0.0))))
            row[f"budget_remaining_{offset}d_before_crash"] = float(sample.get("annual_budget_remaining", 0.0))
            row[f"budget_used_pct_{offset}d_before_crash"] = _safe_ratio(used, limit)
        rows.append(row)
    return rows


def _dynamic_policy_audit_rows(policy: str, cov: pd.DataFrame, trades: pd.DataFrame, lifecycle: pd.DataFrame) -> list[dict[str, Any]]:
    non_valid = int((trades.get("quote_quality_status", pd.Series(dtype=str)).fillna("MISSING").astype(str) != "VALID").sum()) if not trades.empty else 0
    after_expiry = int((trades["date"] > trades["expiry"]).sum()) if not trades.empty and "expiry" in trades else 0
    budget_breach = _annual_budget_breach_count(cov)
    target_changed = _target_mapping_changed(cov)
    checks = [
        ("non_VALID_quote_trades_count", non_valid),
        ("trade_date_after_expiry_count", after_expiry),
        ("annual_budget_breach_count", budget_breach),
        ("target_coverage_mapping_changed_count", target_changed),
    ]
    return [
        {"section": "policy_audit", "policy": policy, "check": check, "status": "PASS" if count == 0 else "FAIL", "count": int(count)}
        for check, count in checks
    ]


def _annual_cost_total(trades: pd.DataFrame) -> float:
    if trades.empty:
        return 0.0
    opens = trades[trades["reason"].astype(str).eq("risk_scaled_hedge_open")]
    return float(-opens["cash_flow"].sum()) if not opens.empty else 0.0


def _annual_budget_breach_count(cov: pd.DataFrame) -> int:
    if cov.empty:
        return 0
    breaches = 0
    for _, group in cov.groupby("year"):
        used = pd.to_numeric(group["annual_budget_used"], errors="coerce").fillna(0.0).max()
        limit = pd.to_numeric(group["annual_budget_limit"], errors="coerce").fillna(0.0).max() if "annual_budget_limit" in group else used
        if used > limit + 1e-8:
            breaches += 1
    return int(breaches)


def _target_mapping_changed(cov: pd.DataFrame) -> int:
    if cov.empty:
        return 0
    # Policy variants must not mutate the existing score-to-target columns during a run.
    recomputed = cov["HedgeNeedScore"].map(_target_from_score_for_audit)
    changed = ~np.isclose(pd.to_numeric(cov["target_hedge_coverage"], errors="coerce"), recomputed, rtol=1e-10, atol=1e-10)
    return int(changed.sum())


def _target_from_score_for_audit(score: float) -> float:
    value = float(np.clip(score, 0.0, 100.0)) if pd.notna(score) else 0.0
    if value <= 25.0:
        return _linear(value, 0.0, 25.0, 0.0, 0.10)
    if value <= 50.0:
        return _linear(value, 25.0, 50.0, 0.10, 0.25)
    if value <= 75.0:
        return _linear(value, 50.0, 75.0, 0.25, 0.50)
    return _linear(value, 75.0, 100.0, 0.50, 0.80)


def _linear(value: float, x0: float, x1: float, y0: float, y1: float) -> float:
    if x1 == x0:
        return y1
    return y0 + (value - x0) * (y1 - y0) / (x1 - x0)



def _daily_new_hedge_cost(trades: pd.DataFrame) -> pd.Series:
    if trades.empty or "reason" not in trades:
        return pd.Series(dtype=float)
    opens = trades[trades["reason"].astype(str).eq("risk_scaled_hedge_open")].copy()
    if opens.empty:
        return pd.Series(dtype=float)
    return -opens.groupby(opens["date"].dt.normalize())["cash_flow"].sum()


def _budget_timeline_rows(cov: pd.DataFrame, daily_cost: pd.Series) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in cov.itertuples(index=False):
        date = pd.Timestamp(row.date).normalize()
        used = float(getattr(row, "annual_budget_used", 0.0))
        remaining = float(getattr(row, "annual_budget_remaining", 0.0))
        limit = used + remaining
        rows.append(
            {
                "section": "annual_budget_usage_timeline",
                "date": str(date.date()),
                "year": int(getattr(row, "year")) if pd.notna(getattr(row, "year")) else "",
                "HedgeNeedScore": float(getattr(row, "HedgeNeedScore", np.nan)),
                "target_hedge_coverage": float(getattr(row, "target_hedge_coverage", np.nan)),
                "current_hedge_coverage": float(getattr(row, "current_hedge_coverage_after_entry", np.nan)),
                "hedge_gap": float(getattr(row, "hedge_gap_after_entry", np.nan)),
                "annual_budget_used": used,
                "annual_budget_remaining": remaining,
                "budget_used_pct": _safe_ratio(used, limit),
                "new_hedge_cost_today": float(daily_cost.get(date, 0.0)),
                "active_put_spread_count": int(getattr(row, "active_put_spread_count", 0)),
                "rejection_reason": str(getattr(row, "rejection_reason", "")),
            }
        )
    return rows


def _budget_score_bucket_rows(cov: pd.DataFrame, trades: pd.DataFrame, daily_cost: pd.Series) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if cov.empty:
        return rows
    entry_days = cov[["date", "HedgeNeedScore", "hedge_gap_after_entry", "annual_budget_used", "annual_budget_remaining"]].copy()
    entry_days["date_norm"] = entry_days["date"].dt.normalize()
    entry_days["new_hedge_cost_today"] = entry_days["date_norm"].map(daily_cost).fillna(0.0)
    entry_days["score_bucket"] = entry_days["HedgeNeedScore"].map(_score_bucket)
    total_budget_limit = _total_budget_limit(cov)
    position_bucket = _position_entry_buckets(trades, entry_days)
    crash_contribution = _crash_coverage_contribution_by_bucket(position_bucket)
    entry_counts = _entry_count_by_bucket(trades, position_bucket)
    for bucket in ["0-25", "25-50", "50-75", "75-100"]:
        group = entry_days[entry_days["score_bucket"].eq(bucket)]
        spent = float(group["new_hedge_cost_today"].sum())
        rows.append(
            {
                "section": "budget_usage_by_score_bucket",
                "score_bucket": bucket,
                "hedge_cost_spent": spent,
                "pct_of_total_annual_budget_limit": _safe_ratio(spent, total_budget_limit),
                "number_of_entries": int(entry_counts.get(bucket, 0)),
                "average_hedge_gap": float(pd.to_numeric(group["hedge_gap_after_entry"], errors="coerce").mean()) if not group.empty else np.nan,
                "realized_crash_coverage_contribution_if_available": int(crash_contribution.get(bucket, 0)),
            }
        )
    return rows


def _budget_exhaustion_rows(cov: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for year, group in cov.groupby("year"):
        if pd.isna(year):
            continue
        group = group.sort_values("date").copy()
        used = pd.to_numeric(group["annual_budget_used"], errors="coerce").fillna(0.0)
        remaining = pd.to_numeric(group["annual_budget_remaining"], errors="coerce").fillna(0.0)
        group["budget_used_pct"] = np.where((used + remaining) > 0, used / (used + remaining), np.nan)
        row: dict[str, Any] = {"section": "budget_exhaustion_timing", "year": int(year)}
        for threshold in [0.25, 0.50, 0.75, 1.00]:
            hit = group[group["budget_used_pct"] >= threshold]
            label = f"{int(threshold * 100)}pct"
            if hit.empty:
                row[f"date_budget_{label}_used"] = ""
                row[f"HedgeNeedScore_at_{label}"] = ""
                row[f"target_coverage_at_{label}"] = ""
                continue
            first = hit.iloc[0]
            row[f"date_budget_{label}_used"] = str(pd.Timestamp(first["date"]).date())
            row[f"HedgeNeedScore_at_{label}"] = float(first["HedgeNeedScore"])
            row[f"target_coverage_at_{label}"] = float(first["target_hedge_coverage"])
        row["crash_happened_after_budget_exhaustion"] = _crash_after_budget_exhaustion(int(year), row.get("date_budget_100pct_used", ""))
        rows.append(row)
    return rows


def _budget_crash_pre_window_rows(cov: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, (start_text, _) in CRASH_WINDOWS.items():
        start = pd.Timestamp(start_text)
        for offset in [180, 90, 60, 30]:
            sample_date = start - pd.Timedelta(days=offset)
            prior = cov[cov["date"] <= sample_date].sort_values("date").tail(1)
            if prior.empty:
                rows.append({"section": "crash_pre_window_budget_state", "crash_window": label, "days_before_crash": offset, "status": "WARN", "detail": "NO_PRIOR_ROW"})
                continue
            row = prior.iloc[0]
            used = float(row["annual_budget_used"])
            remaining = float(row["annual_budget_remaining"])
            reason = str(row.get("rejection_reason", ""))
            rows.append(
                {
                    "section": "crash_pre_window_budget_state",
                    "crash_window": label,
                    "days_before_crash": offset,
                    "sample_date": str(pd.Timestamp(row["date"]).date()),
                    "budget_remaining": remaining,
                    "budget_used_pct": _safe_ratio(used, used + remaining),
                    "hedge_gap": float(row["hedge_gap_after_entry"]),
                    "budget_blocked_execution": reason == "BUDGET_EXCEEDED",
                    "quote_quality_blocked_execution": reason == "QUOTE_NOT_VALID",
                    "rejection_reason": reason,
                }
            )
    return rows


def _budget_crash_classification_rows(cov: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, (start_text, _) in CRASH_WINDOWS.items():
        start = pd.Timestamp(start_text)
        pre = cov[(cov["date"] >= start - pd.Timedelta(days=180)) & (cov["date"] < start)].copy()
        if pre.empty:
            rows.append({"section": "budget_diagnostic_classification", "crash_window": label, "classification": "MIXED", "detail": "NO_PRE_WINDOW_ROWS"})
            continue
        classification = _classify_budget_issue(pre)
        counts = pre["rejection_reason"].astype(str).replace("", np.nan).dropna().value_counts().to_dict()
        rows.append(
            {
                "section": "budget_diagnostic_classification",
                "crash_window": label,
                "classification": classification,
                "budget_exceeded_days": int(counts.get("BUDGET_EXCEEDED", 0)),
                "quote_not_valid_days": int(counts.get("QUOTE_NOT_VALID", 0)),
                "no_contract_found_days": int(counts.get("NO_CONTRACT_FOUND", 0)),
                "hedge_gap_too_small_days": int(counts.get("HEDGE_GAP_TOO_SMALL", 0)),
                "average_HedgeNeedScore": float(pd.to_numeric(pre["HedgeNeedScore"], errors="coerce").mean()),
                "average_hedge_gap": float(pd.to_numeric(pre["hedge_gap_after_entry"], errors="coerce").mean()),
                "budget_remaining_before_crash": float(pd.to_numeric(pre["annual_budget_remaining"], errors="coerce").iloc[-1]),
            }
        )
    return rows


def _score_bucket(score: Any) -> str:
    try:
        value = float(score)
    except (TypeError, ValueError):
        return "MISSING"
    if pd.isna(value):
        return "MISSING"
    if value < 25.0:
        return "0-25"
    if value < 50.0:
        return "25-50"
    if value < 75.0:
        return "50-75"
    return "75-100"


def _total_budget_limit(cov: pd.DataFrame) -> float:
    total = 0.0
    for _, group in cov.groupby("year"):
        used = pd.to_numeric(group["annual_budget_used"], errors="coerce").fillna(0.0)
        remaining = pd.to_numeric(group["annual_budget_remaining"], errors="coerce").fillna(0.0)
        total += float((used + remaining).max())
    return total


def _position_entry_buckets(trades: pd.DataFrame, entry_days: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["position_id", "entry_date", "exit_date", "score_bucket"])
    opens = trades[trades["reason"].astype(str).eq("risk_scaled_hedge_open")].copy()
    if opens.empty:
        return pd.DataFrame(columns=["position_id", "entry_date", "exit_date", "score_bucket"])
    entries = opens.groupby("position_id")["date"].min().rename("entry_date").reset_index()
    exits = trades[~trades["reason"].astype(str).eq("risk_scaled_hedge_open")].groupby("position_id")["date"].max().rename("exit_date").reset_index()
    expiries = opens.groupby("position_id")["expiry"].max().rename("expiry").reset_index() if "expiry" in opens else pd.DataFrame(columns=["position_id", "expiry"])
    positions = entries.merge(exits, on="position_id", how="left").merge(expiries, on="position_id", how="left")
    positions["exit_date"] = positions["exit_date"].fillna(positions["expiry"])
    score_lookup = entry_days[["date_norm", "score_bucket"]].drop_duplicates("date_norm")
    positions["date_norm"] = positions["entry_date"].dt.normalize()
    return positions.merge(score_lookup, on="date_norm", how="left")


def _entry_count_by_bucket(trades: pd.DataFrame, position_bucket: pd.DataFrame) -> dict[str, int]:
    if trades.empty or position_bucket.empty:
        return {}
    return position_bucket["score_bucket"].fillna("MISSING").value_counts().astype(int).to_dict()


def _crash_coverage_contribution_by_bucket(position_bucket: pd.DataFrame) -> dict[str, int]:
    out: dict[str, set[pd.Timestamp]] = {}
    if position_bucket.empty:
        return {}
    for _, pos in position_bucket.iterrows():
        bucket = str(pos.get("score_bucket", "MISSING"))
        entry = pd.Timestamp(pos["entry_date"])
        exit_date = pd.Timestamp(pos["exit_date"])
        if pd.isna(entry) or pd.isna(exit_date):
            continue
        for start_text, end_text in CRASH_WINDOWS.values():
            start = pd.Timestamp(start_text)
            end = pd.Timestamp(end_text)
            overlap_start = max(entry, start)
            overlap_end = min(exit_date, end)
            if overlap_start <= overlap_end:
                out.setdefault(bucket, set()).update(pd.date_range(overlap_start, overlap_end, freq="D"))
    return {bucket: len(days) for bucket, days in out.items()}


def _crash_after_budget_exhaustion(year: int, exhaustion_date: Any) -> bool:
    if not exhaustion_date:
        return False
    try:
        exhausted = pd.Timestamp(exhaustion_date)
    except (TypeError, ValueError):
        return False
    for start_text, _ in CRASH_WINDOWS.values():
        start = pd.Timestamp(start_text)
        if start.year == year and start > exhausted:
            return True
    return False


def _classify_budget_issue(pre: pd.DataFrame) -> str:
    reasons = pre["rejection_reason"].astype(str).replace("", np.nan).dropna().value_counts()
    budget_days = int(reasons.get("BUDGET_EXCEEDED", 0))
    quote_days = int(reasons.get("QUOTE_NOT_VALID", 0))
    no_contract_days = int(reasons.get("NO_CONTRACT_FOUND", 0))
    gap_small_days = int(reasons.get("HEDGE_GAP_TOO_SMALL", 0))
    avg_score = float(pd.to_numeric(pre["HedgeNeedScore"], errors="coerce").mean())
    avg_gap = float(pd.to_numeric(pre["hedge_gap_after_entry"], errors="coerce").mean())
    used = pd.to_numeric(pre["annual_budget_used"], errors="coerce").fillna(0.0)
    remaining = pd.to_numeric(pre["annual_budget_remaining"], errors="coerce").fillna(0.0)
    used_pct_end = _safe_ratio(float(used.iloc[-1]), float(used.iloc[-1] + remaining.iloc[-1]))
    used_pct = float(used_pct_end) if used_pct_end != "" else 0.0
    if avg_score < 20.0:
        return "SCORE_TOO_LOW"
    if avg_gap <= DEFAULT_HEDGE_GAP_THRESHOLD:
        return "HEDGE_GAP_TOO_SMALL"
    if budget_days > 0 and used_pct >= 0.95 and budget_days >= max(quote_days, no_contract_days):
        return "BUDGET_SPENT_TOO_EARLY"
    if quote_days > 0 and quote_days >= max(budget_days, no_contract_days):
        return "BUDGET_AVAILABLE_BUT_QUOTES_FAILED"
    if no_contract_days > 0 and no_contract_days >= max(budget_days, quote_days):
        return "BUDGET_AVAILABLE_BUT_NO_CONTRACT"
    if gap_small_days > 0 and gap_small_days >= max(budget_days, quote_days, no_contract_days):
        return "HEDGE_GAP_TOO_SMALL"
    return "MIXED"



def _breakdown_annual_cost(cov: pd.DataFrame, trades: pd.DataFrame, equity: pd.DataFrame, config: dict) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if cov.empty:
        return rows
    opens = trades[trades.get("reason", pd.Series(dtype=str)).astype(str).eq("risk_scaled_hedge_open")].copy() if not trades.empty else pd.DataFrame()
    annual_cost = -opens.groupby("year")["cash_flow"].sum() if not opens.empty else pd.Series(dtype=float)
    first_equity = equity.groupby("year")["total_equity"].first() if not equity.empty and "total_equity" in equity else pd.Series(dtype=float)
    for year, group in cov.groupby("year"):
        if pd.isna(year):
            continue
        year_int = int(year)
        cost = float(annual_cost.get(year_int, 0.0))
        base_equity = float(first_equity.get(year_int, group["stock_equity"].iloc[0] if "stock_equity" in group else np.nan))
        limit = float(pd.to_numeric(group["annual_budget_used"], errors="coerce").fillna(0).max() + pd.to_numeric(group["annual_budget_remaining"], errors="coerce").fillna(0).iloc[-1])
        remaining = float(pd.to_numeric(group["annual_budget_remaining"], errors="coerce").fillna(0).iloc[-1])
        budget_constrained_days = int(group["rejection_reason"].astype(str).eq("BUDGET_EXCEEDED").sum())
        coverage_days = int((pd.to_numeric(group["current_hedge_coverage_after_entry"], errors="coerce").fillna(0) > 0).sum())
        high_cost_low_protection = bool(cost > 0 and coverage_days < max(5, int(len(group) * 0.10)))
        rows.append(
            {
                "section": "annual_cost_breakdown",
                "year": year_int,
                "annual_hedge_cost": cost,
                "annual_hedge_cost_pct_portfolio_equity": _safe_ratio(cost, base_equity),
                "annual_budget_limit": limit,
                "annual_budget_used_pct": _safe_ratio(cost, limit),
                "annual_budget_remaining": remaining,
                "budget_constrained_days": budget_constrained_days,
                "budget_constrained_coverage": budget_constrained_days > 0,
                "high_cost_but_low_crash_protection": high_cost_low_protection,
            }
        )
    return rows


def _breakdown_coverage_gap(cov: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if cov.empty:
        return rows
    for year, group in cov.groupby("year"):
        if pd.isna(year):
            continue
        failed = group[group["attempted_entry_today"].astype(bool) & ~group["entry_success"].astype(bool)]
        top_reason = ""
        if not failed.empty:
            counts = failed["rejection_reason"].astype(str).value_counts()
            top_reason = str(counts.index[0]) if not counts.empty else ""
        rows.append(
            {
                "section": "coverage_gap_breakdown",
                "year": int(year),
                "average_target_coverage": float(pd.to_numeric(group["target_hedge_coverage"], errors="coerce").mean()),
                "average_current_coverage": float(pd.to_numeric(group["current_hedge_coverage_after_entry"], errors="coerce").mean()),
                "average_hedge_gap": float(pd.to_numeric(group["hedge_gap_after_entry"], errors="coerce").mean()),
                "max_hedge_gap": float(pd.to_numeric(group["hedge_gap_after_entry"], errors="coerce").max()),
                "days_target_gt_current": int((pd.to_numeric(group["target_hedge_coverage"], errors="coerce") > pd.to_numeric(group["current_hedge_coverage_after_entry"], errors="coerce")).sum()),
                "days_execution_attempted": int(group["attempted_entry_today"].astype(bool).sum()),
                "days_execution_failed": int(len(failed)),
                "top_rejection_reason": top_reason,
            }
        )
    return rows


def _breakdown_crash_windows(cov: pd.DataFrame, trades: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if cov.empty:
        return rows
    for label, (start_text, end_text) in CRASH_WINDOWS.items():
        start = pd.Timestamp(start_text)
        end = pd.Timestamp(end_text)
        pre = cov[(cov["date"] >= start - pd.Timedelta(days=180)) & (cov["date"] < start)].copy()
        crash = cov[(cov["date"] >= start) & (cov["date"] <= end)].copy()
        if pre.empty:
            rows.append({"section": "crash_window_breakdown", "crash_window": label, "status": "WARN", "detail": "NO_PRE_WINDOW_ROWS"})
            continue
        failed = pre[pre["attempted_entry_today"].astype(bool) & ~pre["entry_success"].astype(bool)]
        top_reasons = _top_reasons(failed)
        score_rose_failed = bool(((pre["HedgeNeedScore"].diff().fillna(0) > 0) & (pre["current_hedge_coverage_after_entry"].diff().fillna(0) <= 0)).any())
        exited_before = _positions_exited_before_crash(trades, start)
        rows.append(
            {
                "section": "crash_window_breakdown",
                "crash_window": label,
                "pre_crash_average_HedgeNeedScore": float(pd.to_numeric(pre["HedgeNeedScore"], errors="coerce").mean()),
                "pre_crash_target_coverage": float(pd.to_numeric(pre["target_hedge_coverage"], errors="coerce").mean()),
                "pre_crash_current_coverage": float(pd.to_numeric(pre["current_hedge_coverage_after_entry"], errors="coerce").mean()),
                "hedge_gap_before_crash": float(pd.to_numeric(pre["hedge_gap_after_entry"], errors="coerce").iloc[-1]),
                "crash_coverage_ratio": float((pd.to_numeric(crash["current_hedge_coverage_after_entry"], errors="coerce").fillna(0) > 0).mean()) if not crash.empty else np.nan,
                "budget_remaining_before_crash": float(pd.to_numeric(pre["annual_budget_remaining"], errors="coerce").iloc[-1]),
                "top_rejection_reasons_before_crash": top_reasons,
                "score_rose_but_execution_failed": score_rose_failed,
                "execution_succeeded_but_position_exited_before_crash": exited_before,
            }
        )
    return rows


def _breakdown_forced_exits(cov: pd.DataFrame, trades: pd.DataFrame, lifecycle: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if lifecycle.empty or trades.empty:
        return rows
    forced = lifecycle[lifecycle.get("issue", pd.Series(dtype=str)).astype(str).eq("forced_unfilled_exit")].copy()
    if forced.empty:
        return rows
    forced["exit_date"] = pd.to_datetime(forced["exit_date"], errors="coerce")
    forced["year"] = forced["exit_date"].dt.year
    entry = _entry_position_metadata(trades)
    forced = forced.merge(entry, on="position_id", how="left")
    if not cov.empty:
        score_by_date = cov[["date", "HedgeNeedScore"]].copy()
        forced = forced.merge(score_by_date, left_on="exit_date", right_on="date", how="left", suffixes=("", "_coverage"))
    for year, group in forced.groupby("year"):
        rows.append({"section": "forced_exit_breakdown", "dimension": "year", "bucket": int(year) if pd.notna(year) else "", "forced_unfilled_exit_count": int(len(group))})
    for bucket, group in forced.groupby(forced["long_put_moneyness"].map(_moneyness_bucket)):
        rows.append({"section": "forced_exit_breakdown", "dimension": "moneyness", "bucket": bucket, "forced_unfilled_exit_count": int(len(group))})
    for bucket, group in forced.groupby(forced["entry_dte"].map(_entry_dte_bucket)):
        rows.append({"section": "forced_exit_breakdown", "dimension": "entry_dte", "bucket": bucket, "forced_unfilled_exit_count": int(len(group))})
    for reason, group in forced.groupby(forced.get("issue", pd.Series(dtype=str)).astype(str)):
        rows.append({"section": "forced_exit_breakdown", "dimension": "exit_reason", "bucket": reason, "forced_unfilled_exit_count": int(len(group))})
    high_score_count = int((pd.to_numeric(forced.get("HedgeNeedScore", pd.Series(dtype=float)), errors="coerce") >= 50.0).sum())
    rows.append(
        {
            "section": "forced_exit_breakdown",
            "dimension": "high_score_period",
            "bucket": "HedgeNeedScore>=50",
            "forced_unfilled_exit_count": high_score_count,
            "total_forced_unfilled_exit_count": int(len(forced)),
        }
    )
    return rows


def _breakdown_rejections(cov: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if cov.empty:
        return rows
    target_reasons = {"QUOTE_NOT_VALID", "BUDGET_EXCEEDED", "NO_CONTRACT_FOUND", "LOW_LIQUIDITY"}
    rejected = cov[cov["rejection_reason"].astype(str).isin(target_reasons)].copy()
    for reason, group in rejected.groupby("rejection_reason"):
        affected_years = ",".join(str(int(year)) for year in sorted(group["year"].dropna().unique()))
        affected_crash_windows = _affected_crash_windows(group)
        rows.append(
            {
                "section": "rejection_reason_deep_dive",
                "rejection_reason": reason,
                "affected_years": affected_years,
                "affected_crash_windows": affected_crash_windows,
                "rejected_days": int(len(group)),
                "average_HedgeNeedScore_on_rejected_days": float(pd.to_numeric(group["HedgeNeedScore"], errors="coerce").mean()),
                "average_hedge_gap_on_rejected_days": float(pd.to_numeric(group["hedge_gap"], errors="coerce").mean()),
                "days_target_coverage_gt_0": int((pd.to_numeric(group["target_hedge_coverage"], errors="coerce") > 0).sum()),
            }
        )
    return rows


def _entry_position_metadata(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["position_id"])
    opens = trades[trades["reason"].astype(str).eq("risk_scaled_hedge_open")].copy()
    rows: list[dict[str, Any]] = []
    for pos_id, group in opens.groupby("position_id"):
        buys = group[(group["action"] == "BUY") & (group["cp"] == "P")]
        sells = group[(group["action"] == "SELL") & (group["cp"] == "P")]
        if buys.empty or sells.empty:
            continue
        buy = buys.iloc[0]
        sell = sells.iloc[0]
        underlying = float(buy.get("txf_close_at_trade", np.nan))
        rows.append(
            {
                "position_id": pos_id,
                "entry_date": buy["date"],
                "entry_dte": float(buy.get("dte_at_trade", np.nan)),
                "long_put_moneyness": _safe_ratio(float(buy["strike"]), underlying),
                "short_put_moneyness": _safe_ratio(float(sell["strike"]), underlying),
            }
        )
    return pd.DataFrame(rows)


def _positions_exited_before_crash(trades: pd.DataFrame, crash_start: pd.Timestamp) -> bool:
    if trades.empty:
        return False
    opens = trades[trades["reason"].astype(str).eq("risk_scaled_hedge_open")].copy()
    exits = trades[~trades["reason"].astype(str).eq("risk_scaled_hedge_open")].copy()
    if opens.empty or exits.empty:
        return False
    entry_dates = opens.groupby("position_id")["date"].min()
    exit_dates = exits.groupby("position_id")["date"].max()
    for pos_id, entry_date in entry_dates.items():
        exit_date = exit_dates.get(pos_id)
        if pd.notna(entry_date) and pd.notna(exit_date) and entry_date < crash_start and exit_date < crash_start:
            return True
    return False


def _top_reasons(df: pd.DataFrame) -> str:
    if df.empty or "rejection_reason" not in df:
        return ""
    counts = df["rejection_reason"].astype(str).replace("", np.nan).dropna().value_counts().head(3)
    return ";".join(f"{reason}:{int(count)}" for reason, count in counts.items())


def _affected_crash_windows(group: pd.DataFrame) -> str:
    labels: list[str] = []
    for label, (start_text, end_text) in CRASH_WINDOWS.items():
        start = pd.Timestamp(start_text)
        end = pd.Timestamp(end_text)
        pre_start = start - pd.Timedelta(days=180)
        if ((group["date"] >= pre_start) & (group["date"] <= end)).any():
            labels.append(label)
    return ",".join(labels)


def _moneyness_bucket(value: Any) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "UNKNOWN"
    if pd.isna(val):
        return "UNKNOWN"
    if val < 0.88:
        return "<0.88"
    if val < 0.91:
        return "0.88-0.91"
    if val < 0.94:
        return "0.91-0.94"
    return ">=0.94"


def _entry_dte_bucket(value: Any) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "UNKNOWN"
    if pd.isna(val):
        return "UNKNOWN"
    if val <= 60:
        return "<=60"
    if val <= 90:
        return "61-90"
    if val <= 120:
        return "91-120"
    return ">120"


def _summary_rows(coverage: pd.DataFrame, trades: pd.DataFrame, lifecycle: pd.DataFrame, equity: pd.DataFrame, config: dict) -> list[dict[str, Any]]:
    forced_count = int((lifecycle.get("issue", pd.Series(dtype=str)).astype(str) == "forced_unfilled_exit").sum()) if not lifecycle.empty else 0
    rows = [
        {"section": "summary", "metric": "average_target_coverage", "value": _mean(coverage, "target_hedge_coverage")},
        {"section": "summary", "metric": "average_current_coverage", "value": _mean(coverage, "current_hedge_coverage_after_entry")},
        {"section": "summary", "metric": "average_hedge_gap", "value": _mean(coverage, "hedge_gap_after_entry")},
        {"section": "summary", "metric": "forced_unfilled_exit_count", "value": forced_count},
        {"section": "summary", "metric": "trades_count", "value": int(len(trades))},
    ]
    if not coverage.empty and "score_confidence" in coverage:
        for label, count in coverage["score_confidence"].fillna("MISSING").astype(str).value_counts().items():
            rows.append({"section": "score_confidence_distribution", "metric": label, "value": int(count)})
    if not coverage.empty and "rejection_reason" in coverage:
        for reason, count in coverage.loc[coverage["rejection_reason"].astype(str).ne(""), "rejection_reason"].value_counts().items():
            rows.append({"section": "rejection_reason_summary", "metric": reason, "value": int(count)})
    failed = coverage[(coverage.get("target_hedge_coverage", pd.Series(dtype=float)).diff().fillna(0) > 0) & (~coverage.get("entry_success", pd.Series(dtype=bool)).astype(bool))]
    rows.append({"section": "summary", "metric": "days_target_rose_but_execution_failed", "value": int(len(failed))})
    return rows


def _crash_coverage_rows(coverage: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if coverage.empty:
        return rows
    df = coverage.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for label, (start_text, end_text) in CRASH_WINDOWS.items():
        start = pd.Timestamp(start_text)
        end = pd.Timestamp(end_text)
        window = df[(df["date"] >= start) & (df["date"] <= end)]
        pre = df[(df["date"] >= start - pd.Timedelta(days=180)) & (df["date"] < start)]
        coverage_ratio = float((window["current_hedge_coverage_after_entry"] > 0).mean()) if not window.empty else np.nan
        failed_follow = int(((pre["target_hedge_coverage"].diff().fillna(0) > 0) & (pre["current_hedge_coverage_after_entry"].diff().fillna(0) <= 0)).sum()) if not pre.empty else 0
        rows.append(
            {
                "section": "crash_window_coverage",
                "crash_window": label,
                "coverage_ratio": coverage_ratio,
                "covered_days": int((window["current_hedge_coverage_after_entry"] > 0).sum()) if not window.empty else 0,
                "window_days": int(len(window)),
                "days_score_rose_before_crash_but_coverage_failed_to_follow": failed_follow,
            }
        )
    return rows


def _annual_budget_rows(trades: pd.DataFrame, equity: pd.DataFrame, config: dict) -> list[dict[str, Any]]:
    if trades.empty:
        return []
    t = trades.copy()
    t["date"] = pd.to_datetime(t["date"], errors="coerce")
    opens = t[t["reason"].astype(str).eq("risk_scaled_hedge_open")]
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
        rows.append(
            {
                "section": "annual_budget_usage",
                "year": int(year),
                "annual_hedge_cost": float(cost),
                "annual_budget_cap": budget_cap,
                "budget_usage_ratio": _safe_ratio(float(cost), budget_cap),
                "annual_budget_breach": bool(pd.notna(budget_cap) and float(cost) > budget_cap + 1e-8),
            }
        )
    return rows


def _audit_rows(trades: pd.DataFrame, coverage: pd.DataFrame, config: dict) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    non_valid = 0
    after_expiry = 0
    close_fill = 0
    if not trades.empty:
        non_valid = int((trades.get("quote_quality_status", pd.Series(dtype=str)).fillna("MISSING").astype(str) != "VALID").sum())
        after_expiry = int((pd.to_datetime(trades["date"], errors="coerce") > pd.to_datetime(trades["expiry"], errors="coerce")).sum())
        expected_buy = trades["option_ask_at_trade"] * (1.0 + trades["slippage_pct_used"])
        expected_sell = (trades["option_bid_at_trade"] * (1.0 - trades["slippage_pct_used"])).clip(lower=0.0)
        buy_bad = trades["action"].eq("BUY") & ~np.isclose(trades["price"], expected_buy, rtol=1e-9, atol=1e-9)
        sell_bad = trades["action"].eq("SELL") & ~np.isclose(trades["price"], expected_sell, rtol=1e-9, atol=1e-9)
        close_fill = int((buy_bad | sell_bad).sum())
    budget_breach = 0
    if not coverage.empty:
        by_year = coverage.copy()
        by_year["date"] = pd.to_datetime(by_year["date"], errors="coerce")
        for _, group in by_year.groupby(by_year["date"].dt.year):
            max_used = pd.to_numeric(group["annual_budget_used"], errors="coerce").max()
            max_cap = max_used + pd.to_numeric(group["annual_budget_remaining"], errors="coerce").iloc[-1]
            if pd.notna(max_used) and pd.notna(max_cap) and max_used > max_cap + 1e-8:
                budget_breach += 1
    checks = [
        ("non_VALID_quote_trades_count", non_valid),
        ("trade_date_after_expiry_count", after_expiry),
        ("annual_budget_breach_count", budget_breach),
        ("close_based_execution_mismatch_count", close_fill),
    ]
    for check, count in checks:
        rows.append({"section": "audit", "check": check, "status": "PASS" if count == 0 else "FAIL", "count": int(count)})
    return rows


def _markdown(summary: pd.DataFrame, coverage: pd.DataFrame, trades: pd.DataFrame) -> str:
    lines = [
        "# Risk-Scaled Hedge Simulation",
        "",
        "Diagnostics-only simulation of HedgeNeedScore, target hedge coverage, hedge gap, and executable put-spread attempts.",
        "",
        "This report does not change score weights, target mapping, strategy parameters, or bid/ask execution rules.",
        "",
        "## Summary",
    ]
    if not summary.empty:
        for _, row in summary[summary["section"].eq("summary")].iterrows():
            lines.append(f"- {row.get('metric', '')}: {row.get('value', '')}")
    lines.extend(["", "## Crash Window Coverage"])
    if not summary.empty:
        crash = summary[summary["section"].eq("crash_window_coverage")]
        for _, row in crash.iterrows():
            lines.append(
                f"- {row.get('crash_window')}: coverage_ratio={row.get('coverage_ratio')}, "
                f"covered_days={row.get('covered_days')}, failed_follow_days={row.get('days_score_rose_before_crash_but_coverage_failed_to_follow')}"
            )
    lines.extend(["", "## Audit"])
    if not summary.empty:
        for _, row in summary[summary["section"].eq("audit")].iterrows():
            lines.append(f"- {row.get('check')}: {row.get('status')} count={row.get('count')}")
    lines.extend(
        [
            "",
            "## Notes",
            "- Execution attempts use existing bid/ask plus slippage logic.",
            "- Non-VALID quotes are blocked by the existing selector and liquidity gate.",
            "- This is a diagnostic report, not a performance conclusion or investment decision.",
        ]
    )
    text = "\n".join(lines) + "\n"
    lowered = text.lower()
    if "best" in lowered or "recommended" in lowered:
        raise ValueError("risk-scaled hedge report contains disallowed wording")
    return text


def _breakdown_markdown(breakdown: pd.DataFrame) -> str:
    lines = [
        "# Risk-Scaled Hedge Breakdown",
        "",
        "Diagnostics-only breakdown of cost, coverage gaps, crash windows, forced exits, and rejection reasons.",
        "",
        "This report does not change strategy rules, HedgeNeedScore weights, target mapping, or execution logic.",
        "",
        "## Annual Cost",
    ]
    if not breakdown.empty:
        annual = breakdown[breakdown["section"].eq("annual_cost_breakdown")]
        for _, row in annual.iterrows():
            lines.append(
                f"- {int(row['year'])}: cost={row.get('annual_hedge_cost')}, "
                f"budget_used_pct={row.get('annual_budget_used_pct')}, "
                f"budget_constrained_days={row.get('budget_constrained_days')}"
            )
        lines.extend(["", "## Coverage Gap"])
        gaps = breakdown[breakdown["section"].eq("coverage_gap_breakdown")]
        for _, row in gaps.iterrows():
            lines.append(
                f"- {int(row['year'])}: avg_target={row.get('average_target_coverage')}, "
                f"avg_current={row.get('average_current_coverage')}, "
                f"top_rejection={row.get('top_rejection_reason')}"
            )
        lines.extend(["", "## Crash Windows"])
        crashes = breakdown[breakdown["section"].eq("crash_window_breakdown")]
        for _, row in crashes.iterrows():
            lines.append(
                f"- {row.get('crash_window')}: crash_coverage_ratio={row.get('crash_coverage_ratio')}, "
                f"top_rejections={row.get('top_rejection_reasons_before_crash')}, "
                f"score_rose_failed={row.get('score_rose_but_execution_failed')}"
            )
        lines.extend(["", "## Forced Exits"])
        forced = breakdown[breakdown["section"].eq("forced_exit_breakdown")]
        for _, row in forced.head(20).iterrows():
            lines.append(
                f"- {row.get('dimension')} {row.get('bucket')}: forced_count={row.get('forced_unfilled_exit_count')}"
            )
        lines.extend(["", "## Rejections"])
        rejects = breakdown[breakdown["section"].eq("rejection_reason_deep_dive")]
        for _, row in rejects.iterrows():
            lines.append(
                f"- {row.get('rejection_reason')}: days={row.get('rejected_days')}, "
                f"avg_score={row.get('average_HedgeNeedScore_on_rejected_days')}, "
                f"avg_gap={row.get('average_hedge_gap_on_rejected_days')}"
            )
    lines.extend(
        [
            "",
            "## Notes",
            "- Breakdown rows are diagnostics only.",
            "- No new trades are generated by this report.",
            "- Dirty data and quote/liquidity failures remain visible as failures or rejection reasons.",
        ]
    )
    text = "\n".join(lines) + "\n"
    lowered = text.lower()
    if "best" in lowered or "recommend" in lowered:
        raise ValueError("risk-scaled hedge breakdown contains disallowed wording")
    return text


def _budget_audit_markdown(audit: pd.DataFrame) -> str:
    lines = [
        "# Hedge Budget Allocation Audit",
        "",
        "Diagnostics-only audit of annual hedge budget timing, score-bucket usage, and crash pre-window budget state.",
        "",
        "This report does not change annual budget, target coverage mapping, HedgeNeedScore, or execution logic.",
        "",
        "## Budget Usage By Score Bucket",
    ]
    if not audit.empty:
        buckets = audit[audit["section"].eq("budget_usage_by_score_bucket")]
        for _, row in buckets.iterrows():
            lines.append(
                f"- {row.get('score_bucket')}: cost={row.get('hedge_cost_spent')}, "
                f"entries={row.get('number_of_entries')}, avg_gap={row.get('average_hedge_gap')}"
            )
        lines.extend(["", "## Budget Exhaustion Timing"])
        exhaustion = audit[audit["section"].eq("budget_exhaustion_timing")]
        for _, row in exhaustion.iterrows():
            lines.append(
                f"- {row.get('year')}: 25pct={row.get('date_budget_25pct_used')}, "
                f"50pct={row.get('date_budget_50pct_used')}, "
                f"75pct={row.get('date_budget_75pct_used')}, "
                f"100pct={row.get('date_budget_100pct_used')}"
            )
        lines.extend(["", "## Crash Classification"])
        classifications = audit[audit["section"].eq("budget_diagnostic_classification")]
        for _, row in classifications.iterrows():
            lines.append(
                f"- {row.get('crash_window')}: classification={row.get('classification')}, "
                f"budget_days={row.get('budget_exceeded_days')}, "
                f"quote_days={row.get('quote_not_valid_days')}, "
                f"no_contract_days={row.get('no_contract_found_days')}"
            )
    lines.extend(
        [
            "",
            "## Notes",
            "- This audit describes allocation timing and blocked execution.",
            "- No trades are created by this report.",
            "- Quote quality, budget, and lifecycle failures remain visible.",
        ]
    )
    text = "\n".join(lines) + "\n"
    lowered = text.lower()
    if "best" in lowered or "recommend" in lowered or "should increase" in lowered:
        raise ValueError("hedge budget allocation audit contains disallowed wording")
    return text


def _dynamic_budget_policy_markdown(diagnostic: pd.DataFrame) -> str:
    lines = [
        "# Dynamic Budget Policy Diagnostic",
        "",
        "Diagnostics-only comparison of fixed annual-budget release policies.",
        "",
        "Total annual budget, target coverage mapping, HedgeNeedScore, quote quality gate, and execution logic are unchanged.",
        "",
        "## Policy Summary",
    ]
    if not diagnostic.empty:
        summary = diagnostic[diagnostic["section"].eq("policy_summary")]
        for _, row in summary.iterrows():
            lines.append(
                f"- {row.get('policy')}: cost={row.get('annual_hedge_cost')}, "
                f"avg_gap={row.get('average_hedge_gap')}, "
                f"avg_current={row.get('average_current_coverage')}, "
                f"budget_block_days={row.get('days_execution_blocked_by_budget')}"
            )
        lines.extend(["", "## Score Buckets"])
        buckets = diagnostic[diagnostic["section"].eq("policy_score_bucket")]
        for _, row in buckets.iterrows():
            lines.append(
                f"- {row.get('policy')} {row.get('score_bucket')}: "
                f"cost={row.get('hedge_cost_spent')}, entries={row.get('number_of_entries')}"
            )
        lines.extend(["", "## Crash Windows"])
        crashes = diagnostic[diagnostic["section"].eq("policy_crash_window")]
        for _, row in crashes.iterrows():
            lines.append(
                f"- {row.get('policy')} {row.get('crash_window')}: "
                f"coverage={row.get('crash_coverage_ratio')}, "
                f"remaining_30d={row.get('budget_remaining_30d_before_crash')}"
            )
        lines.extend(["", "## Audit"])
        audits = diagnostic[diagnostic["section"].eq("policy_audit")]
        for _, row in audits.iterrows():
            lines.append(f"- {row.get('policy')} {row.get('check')}: {row.get('status')} count={row.get('count')}")
    lines.extend(
        [
            "",
            "## Notes",
            "- This report compares budget release behavior only.",
            "- It does not change annual budget or target coverage.",
            "- It does not create new strategy rules.",
        ]
    )
    text = "\n".join(lines) + "\n"
    lowered = text.lower()
    if "best" in lowered or "recommend" in lowered:
        raise ValueError("dynamic budget policy report contains disallowed wording")
    return text


def _mean(df: pd.DataFrame, col: str) -> float:
    if df.empty or col not in df:
        return float("nan")
    return float(pd.to_numeric(df[col], errors="coerce").mean())


def _safe_ratio(numerator: float, denominator: float) -> float | str:
    if denominator is None or pd.isna(denominator) or abs(float(denominator)) < 1e-12:
        return ""
    return float(numerator) / float(denominator)
