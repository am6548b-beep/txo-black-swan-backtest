"""Shared dataclasses and enums."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

OptionType = Literal["C", "P"]
Side = Literal["BUY", "SELL"]


class StrategyState(str, Enum):
    NORMAL = "NORMAL"
    HEDGE_ON = "HEDGE_ON"
    PANIC = "PANIC"
    POST_PANIC = "POST_PANIC"
    SHORT_VOL_ON = "SHORT_VOL_ON"
    AI_SUPPLY_DISTORTION = "AI_SUPPLY_DISTORTION"
    BULLWHIP_COLLAPSE = "BULLWHIP_COLLAPSE"
    STAGFLATION_PRESSURE = "STAGFLATION_PRESSURE"
    STAGFLATION_DEMAND_BREAK = "STAGFLATION_DEMAND_BREAK"


@dataclass(frozen=True)
class OptionContract:
    date: str
    expiry: str
    dte: int
    cp: OptionType
    strike: float
    close: float
    bid: float
    ask: float
    volume: float
    open_interest: float
    iv: float | None
    delta: float | None
    underlying: float
    tradable: bool = True
    reason: str = ""


@dataclass
class Leg:
    contract: OptionContract
    quantity: int
    entry_price: float = 0.0
    last_price: float = 0.0

    @property
    def cp(self) -> OptionType:
        return self.contract.cp

    @property
    def strike(self) -> float:
        return self.contract.strike

    @property
    def expiry(self) -> str:
        return self.contract.expiry

    @property
    def dte(self) -> int:
        return self.contract.dte


@dataclass
class Position:
    id: str
    strategy: str
    entry_date: str
    legs: list[Leg]
    quantity: int
    entry_underlying: float
    entry_cost: float
    realized_pnl: float = 0.0
    closed: bool = False
    close_date: str | None = None
    max_loss: float = 0.0
    credit_received: float = 0.0
    batches_filled: int = 1
    notes: dict[str, float | str | bool] = field(default_factory=dict)


@dataclass
class Trade:
    date: str
    position_id: str
    strategy: str
    action: str
    cp: str
    strike: float
    expiry: str
    quantity: int
    price: float
    cash_flow: float
    cost: float
    reason: str


@dataclass
class PortfolioState:
    date: str
    cash: float
    stock_equity: float
    option_value: float
    total_equity: float
    free_cash: float
    required_margin: float
    margin_usage: float
    state: StrategyState
    macro_state: str = "NORMAL"
    supply_stress_index: float = 0.0
    macro_demand_fragility_index: float = 0.0
    combined_risk_score: float = 0.0
    daily_stock_pnl: float = 0.0
    daily_option_pnl: float = 0.0
    warning: str = ""
