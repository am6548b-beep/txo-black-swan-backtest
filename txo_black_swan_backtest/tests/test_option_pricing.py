from __future__ import annotations

from src.option_pricing import bs_delta, implied_vol


def test_black_scholes_delta_reasonable() -> None:
    call_delta = bs_delta(100, 100, 30, 0.01, 0.2, "C")
    put_delta = bs_delta(100, 100, 30, 0.01, 0.2, "P")
    assert 0.45 < call_delta < 0.60
    assert -0.55 < put_delta < -0.40


def test_implied_vol_failure_returns_none() -> None:
    assert implied_vol(-1, 100, 100, 30, 0.01, "C") is None
    assert implied_vol(10_000, 100, 100, 30, 0.01, "P") is None

