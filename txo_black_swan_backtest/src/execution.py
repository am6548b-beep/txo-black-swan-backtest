"""Conservative execution model using bid/ask, costs, and slippage."""

from __future__ import annotations

from dataclasses import dataclass

from .contracts import Leg, OptionContract, Trade


@dataclass(frozen=True)
class Fill:
    price: float
    cash_flow: float
    cost: float
    quantity: int
    commission: float = 0.0
    tax: float = 0.0


def is_liquid(contract: OptionContract, config: dict) -> bool:
    return (
        contract.tradable
        and contract.volume >= float(config.get("wide_spread_volume_threshold", 50))
        and contract.open_interest >= float(config.get("min_open_interest", 100))
        and contract.ask >= contract.bid >= 0
    )


def is_stress_day(market_row, config: dict) -> bool:
    macro_state = str(getattr(market_row, "macro_state", "NORMAL"))
    if macro_state in {"BULLWHIP_COLLAPSE", "STAGFLATION_DEMAND_BREAK"}:
        return True
    pct = getattr(market_row, "vix_percentile_3y", None)
    if pct is not None and pct == pct and pct >= float(config.get("stress_vix_percentile", 80.0)):
        return True
    return bool(getattr(market_row, "event_flag", 0) == 1)


def fill_price(contract: OptionContract, side: str, config: dict, stress: bool = False) -> float:
    """Return conservative executable price in option points."""

    if config.get("debug_mid_fill", False):
        return (contract.bid + contract.ask) / 2.0
    slip = slippage_pct(contract, config, stress)
    if side == "BUY":
        return contract.ask * (1.0 + slip)
    if side == "SELL":
        return max(0.0, contract.bid * (1.0 - slip))
    raise ValueError(f"Unknown side: {side}")


def slippage_pct(contract: OptionContract, config: dict, stress: bool = False) -> float:
    slip = float(config["stress_slippage_pct"] if stress else config["normal_slippage_pct"])
    if contract.volume < float(config.get("wide_spread_volume_threshold", 50)):
        slip = max(slip, float(config.get("stress_slippage_pct", slip)))
    return slip


def trade_contract(
    date: str,
    position_id: str,
    strategy: str,
    contract: OptionContract,
    side: str,
    quantity: int,
    config: dict,
    reason: str,
    stress: bool = False,
) -> tuple[Fill, Trade]:
    """Execute a buy or sell; buys consume cash, sells add cash."""

    if quantity <= 0:
        raise ValueError("quantity must be positive")
    price = fill_price(contract, side, config, stress)
    point_value = float(config["txo_point_value"])
    gross = price * point_value * quantity
    commission = float(config["commission_per_contract_per_side"]) * quantity
    tax = gross * float(config["option_tax_rate_on_premium"])
    cost = commission + tax
    cash_flow = -gross - cost if side == "BUY" else gross - cost
    signed_qty = quantity if side == "BUY" else -quantity
    fill = Fill(price=price, cash_flow=cash_flow, cost=cost, quantity=signed_qty, commission=commission, tax=tax)
    liquidity_flag = "OK" if is_liquid(contract, config) else "LOW_LIQUIDITY"
    trade = Trade(
        date=date,
        position_id=position_id,
        strategy=strategy,
        action=side,
        cp=contract.cp,
        strike=contract.strike,
        expiry=contract.expiry,
        quantity=signed_qty,
        price=price,
        cash_flow=cash_flow,
        cost=cost,
        reason=reason,
        underlying_price_at_trade=contract.underlying,
        txf_close_at_trade=contract.underlying,
        option_bid_at_trade=contract.bid,
        option_ask_at_trade=contract.ask,
        option_close_at_trade=contract.close,
        slippage_pct_used=slippage_pct(contract, config, stress),
        commission_paid=commission,
        tax_paid=tax,
        liquidity_flag=liquidity_flag,
        bid_ask_estimated=contract.bid_ask_estimated,
        iv_estimated=contract.iv_estimated,
        delta_estimated=contract.delta_estimated,
    )
    return fill, trade


def close_leg(
    date: str,
    position_id: str,
    strategy: str,
    leg: Leg,
    contract: OptionContract,
    quantity_to_close: int,
    config: dict,
    reason: str,
    stress: bool = False,
) -> tuple[Fill, Trade]:
    """Close part or all of an existing leg."""

    if quantity_to_close <= 0:
        raise ValueError("quantity_to_close must be positive")
    side = "SELL" if leg.quantity > 0 else "BUY"
    return trade_contract(
        date,
        position_id,
        strategy,
        contract,
        side,
        quantity_to_close,
        config,
        reason,
        stress,
    )
