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


def calculate_hedge_gap(target_hedge_coverage: float, current_hedge_coverage: float) -> float:
    """Return target minus current hedge coverage."""

    return float(target_hedge_coverage) - float(current_hedge_coverage)


class RiskScaledHedgeStateMachine(BlackSwanStateMachine):
    """Put-spread simulation driven only by target hedge coverage gap."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
        record = self._base_coverage_record(row, stock_equity, beta_exposure, target_coverage, current_coverage, hedge_gap, budget_used, budget_remaining)
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
        per_spread_debit = (long_px - short_px) * float(self.config["txo_point_value"])
        per_spread_cost = per_spread_debit + 2 * float(self.config["commission_per_contract_per_side"])
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
            after_coverage = current_put_spread_coverage(
                self._open_positions("put_spread"),
                beta_exposure,
                float(self.config["txo_point_value"]),
            )
            record["annual_budget_used"] = self.annual_hedge_spend[year_key(row.date)]
            record["annual_budget_remaining"] = max(0.0, budget_cap - self.annual_hedge_spend[year_key(row.date)])
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
        return cap, used, max(0.0, cap - used)

    def _base_coverage_record(
        self,
        row,
        stock_equity: float,
        beta_exposure: float,
        target_coverage: float,
        current_coverage: float,
        hedge_gap: float,
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
            "annual_budget_used": budget_used,
            "annual_budget_remaining": budget_remaining,
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

    out.to_csv(report_dir / "risk_scaled_hedge_simulation.csv", index=False)
    trades.to_csv(report_dir / "risk_scaled_hedge_trades.csv", index=False)
    coverage.to_csv(report_dir / "risk_scaled_hedge_coverage.csv", index=False)
    (report_dir / "risk_scaled_hedge_simulation.md").write_text(_markdown(out, coverage, trades), encoding="utf-8")
    return out, trades, coverage


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


def _mean(df: pd.DataFrame, col: str) -> float:
    if df.empty or col not in df:
        return float("nan")
    return float(pd.to_numeric(df[col], errors="coerce").mean())


def _safe_ratio(numerator: float, denominator: float) -> float | str:
    if denominator is None or pd.isna(denominator) or abs(float(denominator)) < 1e-12:
        return ""
    return float(numerator) / float(denominator)
