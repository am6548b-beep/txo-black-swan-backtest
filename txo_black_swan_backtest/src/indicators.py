"""Indicators with no-lookahead defaults."""

from __future__ import annotations

import pandas as pd


def moving_average(series: pd.Series, window: int) -> pd.Series:
    """Rolling moving average using observations available at the row date."""

    return series.rolling(window=window, min_periods=window).mean()


def rolling_return(series: pd.Series, window: int) -> pd.Series:
    """Past rolling return ending at the row date."""

    return series / series.shift(window) - 1.0


def rolling_percentile_past(
    series: pd.Series,
    window: int,
    min_periods: int,
) -> pd.Series:
    """Percentile rank of today's value against values strictly before today."""

    shifted = series.shift(1)

    # pandas raw=False is clearer here and keeps the index for no-lookahead tests.
    def percentile(window_values: pd.Series) -> float:
        today_idx = window_values.index[-1]
        today_value = series.loc[today_idx]
        past = shifted.loc[window_values.index].dropna()
        if len(past) < min_periods or pd.isna(today_value):
            return float("nan")
        return float((past <= today_value).mean() * 100.0)

    return series.rolling(window=window + 1, min_periods=min_periods + 1).apply(
        percentile,
        raw=False,
    )


def add_market_indicators(market: pd.DataFrame) -> pd.DataFrame:
    """Add no-lookahead market indicators required by the strategy."""

    out = market.sort_values("date").copy()
    out["ma200"] = moving_average(out["tx_close"], 200)
    out["ret_126d"] = rolling_return(out["tx_close"], 126)
    out["vix_5ma"] = moving_average(out["vix"], 5)
    out["vix_percentile_3y"] = rolling_percentile_past(out["vix"], 756, 252)
    out["ret_20d"] = rolling_return(out["tx_close"], 20)
    out["rolling_20d_low"] = out["tx_close"].rolling(20, min_periods=20).min()
    out["drawdown_20d_from_high"] = out["tx_close"] / out["tx_close"].rolling(20, min_periods=20).max() - 1.0
    out["future_event_5d"] = (
        out["event_flag"].shift(-1).rolling(5, min_periods=1).max().shift(-4).fillna(0)
    )
    return out
