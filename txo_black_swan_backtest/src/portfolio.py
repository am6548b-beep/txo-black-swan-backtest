"""Portfolio accounting helpers."""

from __future__ import annotations

from .contracts import PortfolioState, StrategyState


def make_state(
    date: str,
    cash: float,
    stock_equity: float,
    option_value: float,
    required_margin: float,
    state: StrategyState,
    previous_stock_equity: float | None = None,
    previous_option_value: float | None = None,
    warning: str = "",
    macro_state: str = "NORMAL",
    supply_stress_index: float = 0.0,
    macro_demand_fragility_index: float = 0.0,
    combined_risk_score: float = 0.0,
) -> PortfolioState:
    total = cash + stock_equity + option_value
    free_cash = cash - required_margin
    margin_usage = required_margin / total if total > 0 else 1.0
    return PortfolioState(
        date=date,
        cash=cash,
        stock_equity=stock_equity,
        option_value=option_value,
        total_equity=total,
        free_cash=free_cash,
        required_margin=required_margin,
        margin_usage=margin_usage,
        state=state,
        macro_state=macro_state,
        supply_stress_index=supply_stress_index,
        macro_demand_fragility_index=macro_demand_fragility_index,
        combined_risk_score=combined_risk_score,
        daily_stock_pnl=0.0 if previous_stock_equity is None else stock_equity - previous_stock_equity,
        daily_option_pnl=0.0 if previous_option_value is None else option_value - previous_option_value,
        warning=warning,
    )
