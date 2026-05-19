from __future__ import annotations

from src.contracts import OptionContract
from src.execution import fill_price, trade_contract


def _contract() -> OptionContract:
    return OptionContract(
        date="2024-01-02",
        expiry="2024-02-01",
        dte=30,
        cp="P",
        strike=9000,
        close=100,
        bid=95,
        ask=105,
        volume=100,
        open_interest=200,
        iv=0.2,
        delta=-0.2,
        underlying=10000,
    )


def _config() -> dict:
    return {
        "txo_point_value": 50,
        "commission_per_contract_per_side": 30,
        "option_tax_rate_on_premium": 0.001,
        "normal_slippage_pct": 0.03,
        "stress_slippage_pct": 0.12,
        "wide_spread_volume_threshold": 50,
    }


def test_buy_uses_ask_sell_uses_bid() -> None:
    c = _contract()
    cfg = _config()
    assert fill_price(c, "BUY", cfg) == 105 * 1.03
    assert fill_price(c, "SELL", cfg) == 95 * 0.97


def test_costs_are_deducted() -> None:
    c = _contract()
    cfg = _config()
    fill, trade = trade_contract("2024-01-02", "P1", "put_spread", c, "BUY", 2, cfg, "test")
    gross = 105 * 1.03 * 50 * 2
    expected_cost = 30 * 2 + gross * 0.001
    assert fill.cash_flow == -(gross + expected_cost)
    assert trade.cost == expected_cost


def test_stress_slippage_is_larger() -> None:
    c = _contract()
    cfg = _config()
    assert fill_price(c, "BUY", cfg, stress=True) > fill_price(c, "BUY", cfg, stress=False)

