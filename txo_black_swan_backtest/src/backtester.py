"""Backtest orchestration."""

from __future__ import annotations

import pandas as pd

from .data_loader import load_macro_factors, load_market, load_options, load_portfolio
from .indicators import add_market_indicators
from .macro_regime import add_macro_regime_indicators
from .strategies import BlackSwanStateMachine


def run_backtest(data_dir, config: dict, put_params: dict, ic_params: dict, mode: str = "full") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    market = load_market(data_dir)
    if market.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    market = add_market_indicators(market)
    macro = load_macro_factors(data_dir)
    market = add_macro_regime_indicators(market, macro)
    options = load_options(data_dir, market, config)
    portfolio = load_portfolio(data_dir, market, config)
    engine = BlackSwanStateMachine(market, options, portfolio, config, put_params, ic_params, mode=mode)
    equity, trades = engine.run()
    return equity, trades, market
