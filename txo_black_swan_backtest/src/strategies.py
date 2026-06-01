"""Event-driven strategy state machine."""

from __future__ import annotations

import itertools
from dataclasses import asdict

import numpy as np
import pandas as pd

from .contracts import Leg, OptionContract, Position, StrategyState, Trade
from .execution import close_leg, fill_price, is_liquid, is_stress_day, trade_contract
from .macro_regime import macro_restrictions
from .portfolio import make_state
from .risk import can_add_margin, iron_condor_margin, min_dte, short_delta_breached, short_strike_touched
from .utils import year_key


def row_to_contract(row) -> OptionContract:
    return OptionContract(
        date=str(pd.Timestamp(row.date).date()),
        expiry=str(pd.Timestamp(row.expiry).date()),
        dte=int(row.dte),
        cp=str(row.cp),
        strike=float(row.strike),
        close=float(row.close),
        bid=float(row.bid),
        ask=float(row.ask),
        volume=float(row.volume),
        open_interest=float(row.open_interest),
        iv=None if pd.isna(row.iv) else float(row.iv),
        delta=None if pd.isna(row.delta) else float(row.delta),
        underlying=float(row.underlying),
        tradable=bool(row.tradable),
        reason=str(row.reason) if hasattr(row, "reason") else "",
        bid_ask_estimated=bool(getattr(row, "bid_ask_estimated", False)),
        iv_estimated=bool(getattr(row, "iv_estimated", False)),
        delta_estimated=bool(getattr(row, "delta_estimated", False)),
        quote_quality_status=str(getattr(row, "quote_quality_status", "UNKNOWN")),
        spread_pct=None if not hasattr(row, "spread_pct") or pd.isna(getattr(row, "spread_pct")) else float(getattr(row, "spread_pct")),
        is_tradable_quote=bool(getattr(row, "is_tradable_quote", True)),
    )


class ContractSelector:
    """Select contracts using only the current day's option chain."""

    def __init__(self, options: pd.DataFrame, config: dict):
        self.options = options.copy()
        self.config = config
        self._chains_by_date = {
            pd.Timestamp(date): group.copy()
            for date, group in self.options.groupby("date", sort=False)
        } if not self.options.empty and "date" in self.options else {}

    def chain(self, date: pd.Timestamp) -> pd.DataFrame:
        if self.options.empty:
            return self.options
        return self._chains_by_date.get(pd.Timestamp(date), self.options.iloc[0:0])

    def by_key(self, date: pd.Timestamp, expiry: str, cp: str, strike: float) -> OptionContract | None:
        chain = self.chain(date)
        if chain.empty:
            return None
        expiry_ts = pd.Timestamp(expiry)
        rows = chain[
            (chain["expiry"] == expiry_ts)
            & (chain["cp"] == cp)
            & (np.isclose(chain["strike"], strike))
        ]
        if rows.empty:
            return None
        contract = row_to_contract(rows.iloc[0])
        return contract if contract.tradable else None

    def nearest_strike(
        self,
        date: pd.Timestamp,
        cp: str,
        target_strike: float,
        dte_min: int,
        dte_max: int,
        expiry: pd.Timestamp | None = None,
    ) -> OptionContract | None:
        chain = self.chain(date)
        if chain.empty:
            return None
        pool = chain[
            (chain["cp"] == cp)
            & (chain["dte"].between(dte_min, dte_max))
            & (chain["tradable"])
        ].copy()
        if expiry is not None:
            pool = pool[pool["expiry"] == expiry]
        if pool.empty:
            return None
        pool["expiry_score"] = (pool["dte"] - (dte_min + dte_max) / 2.0).abs()
        pool["strike_score"] = (pool["strike"] - target_strike).abs()
        pool = pool.sort_values(["expiry_score", "strike_score"])
        for row in pool.itertuples(index=False):
            contract = row_to_contract(row)
            if is_liquid(contract, self.config):
                return contract
        return None

    def by_delta(
        self,
        date: pd.Timestamp,
        cp: str,
        target_delta: float,
        dte_min: int,
        dte_max: int,
        expiry: pd.Timestamp | None = None,
    ) -> OptionContract | None:
        chain = self.chain(date)
        if chain.empty:
            return None
        pool = chain[
            (chain["cp"] == cp)
            & (chain["dte"].between(dte_min, dte_max))
            & (chain["tradable"])
            & (chain["delta"].notna())
        ].copy()
        if expiry is not None:
            pool = pool[pool["expiry"] == expiry]
        if pool.empty:
            return None
        pool["expiry_score"] = (pool["dte"] - (dte_min + dte_max) / 2.0).abs()
        pool["delta_score"] = (pool["delta"] - target_delta).abs()
        pool = pool.sort_values(["expiry_score", "delta_score"])
        for row in pool.itertuples(index=False):
            contract = row_to_contract(row)
            if is_liquid(contract, self.config):
                return contract
        return None


class BlackSwanStateMachine:
    """Conservative two-stage put-spread plus iron-condor engine."""

    def __init__(
        self,
        market: pd.DataFrame,
        options: pd.DataFrame,
        portfolio: pd.DataFrame,
        config: dict,
        put_params: dict,
        ic_params: dict,
        mode: str = "full",
    ):
        self.market = market.reset_index(drop=True)
        self.options = options
        self.portfolio = portfolio
        self.config = config
        self.put_params = put_params.copy()
        self.ic_params = ic_params.copy()
        self.mode = mode
        self.selector = ContractSelector(options, config)
        self.state = StrategyState.NORMAL
        self.cash = float(config["initial_cash"])
        self.positions: list[Position] = []
        self.trades: list[Trade] = []
        self.states = []
        self.lifecycle_events: list[dict[str, str]] = []
        self.annual_hedge_spend: dict[int, float] = {}
        self.realized_hedge_profit = 0.0
        self.position_counter = itertools.count(1)
        self.last_panic_low: float | None = None

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        prev_stock = None
        prev_option_value = None
        for row in self.market.itertuples(index=False):
            date = pd.Timestamp(row.date)
            expiry_warning = self._expire_past_due_positions(date)
            stock_row = self.portfolio[self.portfolio["date"] == date]
            stock_equity = (
                float(stock_row.iloc[0]["stock_equity"])
                if not stock_row.empty
                else float(self.config["initial_stock_equity"])
            )
            self._refresh_open_legs(date)
            option_value = self._option_liquidation_value(row)
            required_margin = self._required_margin()
            warning = self._risk_update(row, required_margin, option_value, stock_equity)
            if expiry_warning:
                warning = ";".join(filter(None, [warning, expiry_warning]))
            self._check_exits(row)
            if self.mode in {"full", "put_spread_only"}:
                self._check_put_entry(row, stock_equity)
            if self.mode in {"full", "iron_condor_only"}:
                self._check_ic_entry(row, stock_equity)
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

    def _open_positions(self, strategy: str | None = None) -> list[Position]:
        out = [p for p in self.positions if not p.closed]
        return [p for p in out if p.strategy == strategy] if strategy else out

    def _refresh_open_legs(self, date: pd.Timestamp) -> None:
        for pos in self._open_positions():
            refreshed: list[Leg] = []
            for leg in pos.legs:
                contract = self.selector.by_key(date, leg.expiry, leg.cp, leg.strike)
                if contract is None:
                    refreshed.append(leg)
                else:
                    leg.contract = contract
                    refreshed.append(leg)
            pos.legs = refreshed

    def _position_expiry(self, pos: Position) -> pd.Timestamp:
        expiries = [pd.Timestamp(leg.expiry) for leg in pos.legs]
        return min(expiries)

    def _position_has_current_quotes(self, pos: Position, date: pd.Timestamp) -> bool:
        current = str(date.date())
        return all(leg.contract.date == current for leg in pos.legs)

    def _expire_past_due_positions(self, date: pd.Timestamp) -> str:
        expired_ids: list[str] = []
        for pos in self._open_positions():
            if date > self._position_expiry(pos):
                pos.closed = True
                pos.close_date = str(date.date())
                pos.notes["lifecycle_status"] = "forced_unfilled_exit"
                pos.notes["issue"] = "expired_position_error"
                pos.quantity = 0
                self.lifecycle_events.append(
                    {
                        "position_id": pos.id,
                        "entry_date": pos.entry_date,
                        "expiry": str(self._position_expiry(pos).date()),
                        "exit_date": str(date.date()),
                        "status": "WARN",
                        "issue": "forced_unfilled_exit",
                    }
                )
                expired_ids.append(pos.id)
        return f"expired_position_error={','.join(expired_ids)}" if expired_ids else ""

    def _option_liquidation_value(self, market_row) -> float:
        stress = is_stress_day(market_row, self.config)
        value = 0.0
        for pos in self._open_positions():
            if not self._position_has_current_quotes(pos, pd.Timestamp(market_row.date)):
                continue
            for leg in pos.legs:
                side = "SELL" if leg.quantity > 0 else "BUY"
                price = fill_price(leg.contract, side, self.config, stress)
                signed = 1 if leg.quantity > 0 else -1
                value += signed * price * float(self.config["txo_point_value"]) * abs(leg.quantity)
        return value

    def _required_margin(self) -> float:
        return sum(iron_condor_margin(p, float(self.config["txo_point_value"])) for p in self._open_positions())

    def _risk_update(self, row, required_margin: float, option_value: float, stock_equity: float) -> str:
        total_equity = self.cash + option_value + stock_equity
        if required_margin > 0 and total_equity > 0:
            usage = required_margin / total_equity
            if usage > float(self.config["max_margin_usage_pct"]):
                return "margin_usage_above_limit"
        return ""

    def _check_put_entry(self, row, stock_equity: float) -> None:
        if self._open_positions("put_spread"):
            return
        if any(pd.isna(x) for x in [row.ma200, row.ret_126d, row.vix_percentile_3y]):
            return
        macro_state = str(getattr(row, "macro_state", "NORMAL"))
        restriction = macro_restrictions(macro_state)
        vix_threshold = 40.0 if macro_state == "STAGFLATION_PRESSURE" and float(getattr(row, "ValuationRiskIndex", 0.0)) > 70.0 else 30.0
        if not (
            row.tx_close > row.ma200
            and row.ret_126d > 0.20
            and row.vix_percentile_3y < vix_threshold
            and int(row.event_flag) == 0
        ):
            return
        year = year_key(row.date)
        budget_cap = float(self.config["max_annual_hedge_budget_pct"]) * float(restriction["hedge_budget_multiplier"]) * stock_equity
        used = self.annual_hedge_spend.get(year, 0.0)
        if used >= budget_cap:
            return

        low_vix = row.vix_percentile_3y < 20.0
        long_m = self.put_params["long_put_moneyness_low_vix"] if low_vix else self.put_params["long_put_moneyness"]
        short_m = self.put_params["short_put_moneyness_low_vix"] if low_vix else self.put_params["short_put_moneyness"]
        dte_min = int(self.put_params["target_dte_min"])
        dte_max = int(self.put_params["target_dte_max"]) + int(restriction["extend_put_dte"])
        date = pd.Timestamp(row.date)
        long_put = self.selector.nearest_strike(date, "P", row.txf_close * long_m, dte_min, dte_max)
        if long_put is None:
            return
        short_put = self.selector.nearest_strike(
            date,
            "P",
            row.txf_close * short_m,
            dte_min,
            dte_max,
            expiry=pd.Timestamp(long_put.expiry),
        )
        if short_put is None or short_put.strike >= long_put.strike:
            return
        stress = is_stress_day(row, self.config)
        long_px = fill_price(long_put, "BUY", self.config, stress)
        short_px = fill_price(short_put, "SELL", self.config, stress)
        per_spread_debit = (long_px - short_px) * float(self.config["txo_point_value"])
        per_spread_cost = per_spread_debit + 2 * float(self.config["commission_per_contract_per_side"])
        if per_spread_cost <= 0:
            return
        notional_contracts = max(1, int((stock_equity * float(self.config["portfolio_beta"])) / (row.txf_close * float(self.config["txo_point_value"])) * 0.5))
        qty = min(notional_contracts, int((budget_cap - used) // per_spread_cost))
        if qty <= 0:
            return
        pos_id = f"PS-{next(self.position_counter)}"
        fill1, tr1 = trade_contract(str(date.date()), pos_id, "put_spread", long_put, "BUY", qty, self.config, "open_put_spread", stress)
        fill2, tr2 = trade_contract(str(date.date()), pos_id, "put_spread", short_put, "SELL", qty, self.config, "open_put_spread", stress)
        self.cash += fill1.cash_flow + fill2.cash_flow
        self.trades.extend([tr1, tr2])
        spend = -(fill1.cash_flow + fill2.cash_flow)
        self.annual_hedge_spend[year] = used + max(0.0, spend)
        pos = Position(
            id=pos_id,
            strategy="put_spread",
            entry_date=str(date.date()),
            legs=[Leg(long_put, qty, fill1.price), Leg(short_put, -qty, fill2.price)],
            quantity=qty,
            entry_underlying=float(row.tx_close),
            entry_cost=spend,
            max_loss=spend,
            notes={"remaining_entry_cost": spend},
        )
        self.positions.append(pos)
        self.state = StrategyState.HEDGE_ON

    def _check_exits(self, row) -> None:
        for pos in list(self._open_positions()):
            if pos.strategy == "put_spread":
                self._check_put_exit(row, pos)
            elif pos.strategy == "iron_condor":
                self._check_ic_exit(row, pos)
        if self.state == StrategyState.HEDGE_ON and not self._open_positions("put_spread"):
            self.state = StrategyState.POST_PANIC if self.realized_hedge_profit > 0 else StrategyState.NORMAL
        if self.state == StrategyState.SHORT_VOL_ON and not self._open_positions("iron_condor"):
            self.state = StrategyState.POST_PANIC

    def _put_spread_value_per_spread(self, pos: Position, row) -> float:
        if not self._position_has_current_quotes(pos, pd.Timestamp(row.date)):
            return 0.0
        stress = is_stress_day(row, self.config)
        total = 0.0
        qty = max(1, abs(pos.quantity))
        for leg in pos.legs:
            side = "SELL" if leg.quantity > 0 else "BUY"
            price = fill_price(leg.contract, side, self.config, stress)
            signed = 1 if leg.quantity > 0 else -1
            total += signed * price * float(self.config["txo_point_value"]) * abs(leg.quantity)
        return total / qty

    def _check_put_exit(self, row, pos: Position) -> None:
        if pos.closed:
            return
        date_ts = pd.Timestamp(row.date)
        if date_ts > self._position_expiry(pos):
            self._expire_past_due_positions(date_ts)
            return
        if not self._position_has_current_quotes(pos, date_ts):
            return
        long_leg = next(l for l in pos.legs if l.quantity > 0)
        short_leg = next(l for l in pos.legs if l.quantity < 0)
        width_value = (long_leg.strike - short_leg.strike) * float(self.config["txo_point_value"])
        spread_value = self._put_spread_value_per_spread(pos, row)
        drop = float(row.tx_close) / pos.entry_underlying - 1.0
        qty_open = abs(pos.quantity)
        reason = ""
        qty_to_close = 0
        if spread_value >= width_value * float(self.put_params["profit_take_2"]) or drop <= float(self.put_params["drop_take_2"]):
            qty_to_close = qty_open
            reason = "put_spread_full_take_profit"
        elif (
            not pos.notes.get("half_closed")
            and (spread_value >= width_value * float(self.put_params["profit_take_1"]) or drop <= float(self.put_params["drop_take_1"]))
        ):
            qty_to_close = max(1, qty_open // 2)
            reason = "put_spread_half_take_profit"
            pos.notes["half_closed"] = True
            self.state = StrategyState.PANIC
        elif min_dte(pos) < int(self.put_params["exit_dte"]):
            qty_to_close = qty_open
            reason = "put_spread_dte_exit"
        elif spread_value <= pos.entry_cost / max(1, pos.quantity) * float(self.put_params["low_value_exit_pct"]) and min_dte(pos) < int(self.put_params["low_value_exit_dte"]):
            qty_to_close = qty_open
            reason = "put_spread_low_value_exit"
        if qty_to_close:
            self._close_position_qty(row, pos, qty_to_close, reason)
            if pos.closed and pos.realized_pnl > 0:
                self.realized_hedge_profit += pos.realized_pnl
                self.last_panic_low = float(row.tx_close)

    def _check_ic_entry(self, row, stock_equity: float) -> None:
        if self._open_positions("iron_condor"):
            return
        macro_state = str(getattr(row, "macro_state", "NORMAL"))
        restriction = macro_restrictions(macro_state)
        if bool(restriction["block_new_ic"]):
            return
        if self.mode == "full" and self.state not in {StrategyState.POST_PANIC, StrategyState.PANIC}:
            return
        needed = [row.drawdown_20d_from_high, row.rolling_20d_low, row.vix_5ma]
        if any(pd.isna(x) for x in needed):
            return
        bounce = float(row.tx_close) / float(row.rolling_20d_low) - 1.0
        recent = self.market[self.market["date"] <= row.date].tail(5)
        no_new_low = len(recent) >= 5 and float(row.tx_close) > float(recent["tx_close"].min())
        if not (
            row.drawdown_20d_from_high <= -0.10
            and bounce >= 0.03
            and row.vix < row.vix_5ma
            and no_new_low
            and int(row.event_flag) == 0
            and int(getattr(row, "future_event_5d", 0)) == 0
        ):
            return
        date = pd.Timestamp(row.date)
        dte_min = int(self.ic_params["target_dte_min"])
        dte_max = int(self.ic_params["target_dte_max"])
        sp = self.selector.by_delta(date, "P", -abs(float(self.ic_params["short_put_delta"])), dte_min, dte_max)
        sc = self.selector.by_delta(date, "C", abs(float(self.ic_params["short_call_delta"])), dte_min, dte_max, expiry=pd.Timestamp(sp.expiry) if sp else None)
        if sp is None or sc is None:
            return
        width = float(self.ic_params["wing_width"])
        lp = self.selector.nearest_strike(date, "P", sp.strike - width, dte_min, dte_max, expiry=pd.Timestamp(sp.expiry))
        lc = self.selector.nearest_strike(date, "C", sc.strike + width, dte_min, dte_max, expiry=pd.Timestamp(sc.expiry))
        if lp is None or lc is None:
            return
        stress = is_stress_day(row, self.config)
        credit_points = (
            fill_price(sp, "SELL", self.config, stress)
            + fill_price(sc, "SELL", self.config, stress)
            - fill_price(lp, "BUY", self.config, stress)
            - fill_price(lc, "BUY", self.config, stress)
        )
        if credit_points <= 0:
            return
        point_value = float(self.config["txo_point_value"])
        max_loss_per = width * point_value - credit_points * point_value
        allowed = min(
            (self.cash + stock_equity) * float(self.ic_params["max_total_asset_loss_pct"]),
            max(0.0, self.realized_hedge_profit) * float(self.ic_params["max_hedge_profit_giveback_pct"]) if self.mode == "full" else (self.cash + stock_equity) * 0.01,
        ) * float(restriction["ic_risk_multiplier"])
        qty = int(allowed // max_loss_per)
        if qty <= 0:
            return
        margin_config = self.config.copy()
        margin_config["min_free_cash_multiplier"] = float(self.config.get("min_free_cash_multiplier", 2.0)) * float(restriction["free_cash_multiplier"])
        if not can_add_margin(self._required_margin(), max_loss_per * qty, self.cash + stock_equity, self.cash, margin_config):
            return
        pos_id = f"IC-{next(self.position_counter)}"
        fills_trades = [
            trade_contract(str(date.date()), pos_id, "iron_condor", lp, "BUY", qty, self.config, "open_iron_condor", stress),
            trade_contract(str(date.date()), pos_id, "iron_condor", sp, "SELL", qty, self.config, "open_iron_condor", stress),
            trade_contract(str(date.date()), pos_id, "iron_condor", sc, "SELL", qty, self.config, "open_iron_condor", stress),
            trade_contract(str(date.date()), pos_id, "iron_condor", lc, "BUY", qty, self.config, "open_iron_condor", stress),
        ]
        for fill, trade in fills_trades:
            self.cash += fill.cash_flow
            self.trades.append(trade)
        credit = sum(fill.cash_flow for fill, _ in fills_trades)
        pos = Position(
            id=pos_id,
            strategy="iron_condor",
            entry_date=str(date.date()),
            legs=[Leg(lp, qty), Leg(sp, -qty), Leg(sc, -qty), Leg(lc, qty)],
            quantity=qty,
            entry_underlying=float(row.tx_close),
            entry_cost=-credit,
            credit_received=max(0.0, credit),
            max_loss=max_loss_per * qty,
            notes={"remaining_credit": max(0.0, credit)},
        )
        self.positions.append(pos)
        self.state = StrategyState.SHORT_VOL_ON

    def _check_ic_exit(self, row, pos: Position) -> None:
        if pos.closed:
            return
        date_ts = pd.Timestamp(row.date)
        if date_ts > self._position_expiry(pos):
            self._expire_past_due_positions(date_ts)
            return
        if not self._position_has_current_quotes(pos, date_ts):
            return
        liquidation = -self._position_liquidation_value(pos, row)
        credit = max(pos.credit_received, 1.0)
        pnl = pos.credit_received - liquidation
        profit_take = float(self.ic_params["high_stress_profit_take_pct"] if is_stress_day(row, self.config) else self.ic_params["profit_take_pct"])
        reason = ""
        qty = 0
        if pnl >= credit * profit_take:
            reason = "iron_condor_profit_take"
            qty = abs(pos.quantity)
        elif short_delta_breached(pos, float(self.ic_params["stop_delta"])):
            reason = "iron_condor_delta_stop"
            qty = abs(pos.quantity)
        elif short_strike_touched(pos, float(row.tx_close)):
            reason = "iron_condor_short_strike_touched"
            qty = abs(pos.quantity)
        elif pnl <= -pos.max_loss:
            reason = "iron_condor_max_loss"
            qty = abs(pos.quantity)
        elif min_dte(pos) < int(self.ic_params["exit_dte"]):
            reason = "iron_condor_dte_exit"
            qty = abs(pos.quantity)
        elif short_delta_breached(pos, float(self.ic_params["reduce_delta"])) and not pos.notes.get("delta_reduced"):
            reason = "iron_condor_delta_reduce"
            qty = max(1, abs(pos.quantity) // 2)
            pos.notes["delta_reduced"] = True
        if qty:
            self._close_position_qty(row, pos, qty, reason)

    def _position_liquidation_value(self, pos: Position, row) -> float:
        if not self._position_has_current_quotes(pos, pd.Timestamp(row.date)):
            return 0.0
        stress = is_stress_day(row, self.config)
        value = 0.0
        for leg in pos.legs:
            side = "SELL" if leg.quantity > 0 else "BUY"
            price = fill_price(leg.contract, side, self.config, stress)
            signed = 1 if leg.quantity > 0 else -1
            value += signed * price * float(self.config["txo_point_value"]) * abs(leg.quantity)
        return value

    def _close_position_qty(self, row, pos: Position, qty_to_close: int, reason: str) -> None:
        stress = is_stress_day(row, self.config)
        date = str(pd.Timestamp(row.date).date())
        if pd.Timestamp(date) > self._position_expiry(pos):
            self._expire_past_due_positions(pd.Timestamp(date))
            return
        if not self._position_has_current_quotes(pos, pd.Timestamp(date)):
            return
        cash_before = self.cash
        open_qty_before = max(1, abs(pos.quantity))
        for leg in pos.legs:
            fill, trade = close_leg(date, pos.id, pos.strategy, leg, leg.contract, qty_to_close, self.config, reason, stress)
            self.cash += fill.cash_flow
            self.trades.append(trade)
            if leg.quantity > 0:
                leg.quantity -= qty_to_close
            else:
                leg.quantity += qty_to_close
        pos.quantity -= qty_to_close
        realized_cash = self.cash - cash_before
        if pos.strategy == "put_spread":
            remaining_entry = float(pos.notes.get("remaining_entry_cost", pos.entry_cost))
            entry_slice = remaining_entry * qty_to_close / open_qty_before
            pos.notes["remaining_entry_cost"] = remaining_entry - entry_slice
            pos.entry_cost = remaining_entry - entry_slice
            pos.realized_pnl += realized_cash - entry_slice
        else:
            remaining_credit = float(pos.notes.get("remaining_credit", pos.credit_received))
            credit_slice = remaining_credit * qty_to_close / open_qty_before
            pos.notes["remaining_credit"] = remaining_credit - credit_slice
            pos.credit_received = remaining_credit - credit_slice
            pos.max_loss = pos.max_loss * max(0, open_qty_before - qty_to_close) / open_qty_before
            pos.realized_pnl += realized_cash + credit_slice
        if abs(pos.quantity) <= 0:
            pos.closed = True
            pos.close_date = date
