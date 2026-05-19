"""Margin and rule-based risk checks."""

from __future__ import annotations

from .contracts import Position


def iron_condor_margin(position: Position, point_value: float) -> float:
    """Approximate defined-risk margin from the wider wing minus credit."""

    if position.strategy != "iron_condor":
        return 0.0
    puts = sorted([leg.strike for leg in position.legs if leg.cp == "P"])
    calls = sorted([leg.strike for leg in position.legs if leg.cp == "C"])
    if len(puts) < 2 or len(calls) < 2:
        return position.max_loss
    put_width = abs(puts[-1] - puts[0])
    call_width = abs(calls[-1] - calls[0])
    width = max(put_width, call_width)
    gross = width * point_value * abs(position.quantity)
    return max(0.0, gross - position.credit_received)


def can_add_margin(
    current_margin: float,
    additional_margin: float,
    total_equity: float,
    free_cash: float,
    config: dict,
) -> bool:
    if additional_margin <= 0:
        return True
    max_usage = float(config.get("max_margin_usage_pct", 0.35))
    min_cash_mult = float(config.get("min_free_cash_multiplier", 2.0))
    new_margin = current_margin + additional_margin
    usage = new_margin / total_equity if total_equity > 0 else 1.0
    return usage <= max_usage and free_cash >= additional_margin * min_cash_mult


def short_delta_breached(position: Position, threshold: float) -> bool:
    for leg in position.legs:
        if leg.quantity < 0 and leg.contract.delta is not None:
            if abs(float(leg.contract.delta)) > threshold:
                return True
    return False


def short_strike_touched(position: Position, underlying: float) -> bool:
    for leg in position.legs:
        if leg.quantity < 0 and leg.cp == "P" and underlying <= leg.strike:
            return True
        if leg.quantity < 0 and leg.cp == "C" and underlying >= leg.strike:
            return True
    return False


def min_dte(position: Position) -> int:
    return min((leg.contract.dte for leg in position.legs), default=999)

