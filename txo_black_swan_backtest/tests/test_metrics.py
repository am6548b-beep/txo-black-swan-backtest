from __future__ import annotations

import pandas as pd
import pytest

from src.metrics import avg_win_loss, cagr, max_consecutive_losses, max_drawdown, worst_month


def test_max_drawdown() -> None:
    assert max_drawdown(pd.Series([100, 120, 90, 130])) == -0.25


def test_cagr_positive() -> None:
    dates = pd.Series(pd.to_datetime(["2020-01-01", "2021-01-01"]))
    value = cagr(pd.Series([100.0, 110.0]), dates)
    assert 0.09 < value < 0.11


def test_worst_month() -> None:
    eq = pd.DataFrame({"date": pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"]), "total_equity": [100, 90, 99]})
    assert worst_month(eq) == pytest.approx(-0.10)


def test_average_win_loss_and_streak() -> None:
    trades = pd.DataFrame(
        {
            "position_id": ["a", "b", "c", "d"],
            "strategy": ["x"] * 4,
            "cash_flow": [10, -5, -3, 2],
        }
    )
    avg_win, avg_loss = avg_win_loss(trades)
    assert avg_win == 6
    assert avg_loss == -4
    assert max_consecutive_losses(trades) == 2
