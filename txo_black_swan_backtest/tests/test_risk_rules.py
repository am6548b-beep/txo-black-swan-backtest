from __future__ import annotations

from src.contracts import Leg, OptionContract, Position
from src.risk import can_add_margin, min_dte, short_strike_touched


def _c(cp: str, strike: float, delta: float, dte: int = 30) -> OptionContract:
    return OptionContract("2024-01-02", "2024-02-01", dte, cp, strike, 10, 9, 11, 100, 200, 0.2, delta, 10000)


def test_ic_touching_short_strike_forces_exit_condition() -> None:
    pos = Position("IC-1", "iron_condor", "2024-01-02", [_c_leg("P", 9500, -0.05, -1), _c_leg("C", 11000, 0.15, -1)], 1, 10000, 0)
    assert short_strike_touched(pos, 9400)
    assert short_strike_touched(pos, 11100)


def test_dte_threshold_detectable() -> None:
    pos = Position("P1", "put_spread", "2024-01-02", [_c_leg("P", 9000, -0.2, 1, dte=9)], 1, 10000, 100)
    assert min_dte(pos) < 10


def test_margin_usage_blocks_add() -> None:
    assert not can_add_margin(300_000, 100_000, 1_000_000, 50_000, {"max_margin_usage_pct": 0.35, "min_free_cash_multiplier": 2.0})


def test_no_rolling_is_represented_by_full_reentry_rule() -> None:
    # There is intentionally no rolling helper in the risk module; strategy exits first.
    import src.risk as risk

    assert not hasattr(risk, "roll_losing_side")


def _c_leg(cp: str, strike: float, delta: float, qty: int, dte: int = 30) -> Leg:
    return Leg(_c(cp, strike, delta, dte), qty)

