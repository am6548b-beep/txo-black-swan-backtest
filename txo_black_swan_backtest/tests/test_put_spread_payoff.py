from __future__ import annotations

from src.option_pricing import put_spread_expiry_payoff


def test_deterministic_put_spread_payoff() -> None:
    f0 = 40_000
    long_put = 36_000
    short_put = 30_000
    point_value = 50

    assert f0 == 40_000
    assert put_spread_expiry_payoff(32_000, long_put, short_put, point_value) == 200_000
    assert put_spread_expiry_payoff(28_000, long_put, short_put, point_value) == 300_000
