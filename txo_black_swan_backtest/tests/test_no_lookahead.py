from __future__ import annotations

import pandas as pd

from src.indicators import add_market_indicators, moving_average, rolling_percentile_past


def test_vix_percentile_uses_only_prior_data() -> None:
    s = pd.Series([10.0] * 252 + [20.0, 99.0])
    pct = rolling_percentile_past(s, window=756, min_periods=252)
    assert pct.iloc[252] == 100.0
    s_changed_future = s.copy()
    s_changed_future.iloc[253] = 1_000.0
    pct_changed = rolling_percentile_past(s_changed_future, window=756, min_periods=252)
    assert pct_changed.iloc[252] == pct.iloc[252]


def test_ma_does_not_use_future_data() -> None:
    s = pd.Series(range(1, 202), dtype=float)
    ma = moving_average(s, 200)
    assert pd.isna(ma.iloc[198])
    assert ma.iloc[199] == sum(range(1, 201)) / 200
    changed = s.copy()
    changed.iloc[200] = 1_000_000
    ma_changed = moving_average(changed, 200)
    assert ma_changed.iloc[199] == ma.iloc[199]


def test_entry_indicators_do_not_need_next_day_price() -> None:
    market = pd.DataFrame(
        {
            "date": pd.bdate_range("2020-01-01", periods=260),
            "tx_close": list(range(100, 360)),
            "tx_open": list(range(100, 360)),
            "tx_high": list(range(101, 361)),
            "tx_low": list(range(99, 359)),
            "txf_close": list(range(100, 360)),
            "volume": 1,
            "vix": [15.0] * 260,
            "event_flag": 0,
        }
    )
    out = add_market_indicators(market)
    before = out.loc[258, ["ma200", "ret_126d", "vix_percentile_3y"]].copy()
    market.loc[259, "tx_close"] = 1
    changed = add_market_indicators(market)
    after = changed.loc[258, ["ma200", "ret_126d", "vix_percentile_3y"]]
    assert before.equals(after)

