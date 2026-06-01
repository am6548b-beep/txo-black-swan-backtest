from __future__ import annotations

import pandas as pd

from src.hedge_need import (
    hedge_coverage_timeline,
    hedge_gap_audit,
    hedge_need_score_frame,
    hedge_need_score_attribution,
    target_hedge_coverage,
    RISK_WEIGHTS,
    _attribution_markdown,
    _markdown,
)


def test_target_hedge_coverage_mapping() -> None:
    assert target_hedge_coverage(0) == 0.0
    assert 0.09 < target_hedge_coverage(25) <= 0.10
    assert 0.24 < target_hedge_coverage(50) <= 0.25
    assert 0.49 < target_hedge_coverage(75) <= 0.50
    assert 0.79 < target_hedge_coverage(100) <= 0.80


def test_hedge_need_score_uses_fixed_modules() -> None:
    market = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"]),
            "tx_close": [100.0],
            "ma200": [100.0],
            "ret_126d": [0.0],
            "drawdown_20d_from_high": [0.0],
            "vix_percentile_3y": [10.0],
            "ValuationRiskIndex": [80.0],
            "MacroDemandFragilityIndex": [40.0],
            "SupplyStressIndex": [50.0],
            "LiquidityStressIndex": [30.0],
            "vix_is_proxy": [True],
        }
    )
    score = hedge_need_score_frame(market)

    assert "HedgeNeedScore" in score
    assert score.loc[0, "VolatilityComplacencyRisk"] == 90.0
    assert bool(score.loc[0, "vix_proxy_in_use"])


def test_current_hedge_coverage_from_reported_put_spread() -> None:
    score = pd.DataFrame({"date": pd.to_datetime(["2024-01-01"]), "target_hedge_coverage": [0.2]})
    portfolio = pd.DataFrame({"date": pd.to_datetime(["2024-01-01"]), "stock_equity": [1_000_000.0], "portfolio_beta": [1.0]})
    trades = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01"],
            "position_id": ["PS-1", "PS-1"],
            "strategy": ["put_spread", "put_spread"],
            "reason": ["open_put_spread", "open_put_spread"],
            "action": ["BUY", "SELL"],
            "strike": [9000.0, 8000.0],
            "expiry": ["2024-04-17", "2024-04-17"],
            "quantity": [2, 2],
            "cash_flow": [-100.0, 50.0],
        }
    )
    coverage = hedge_coverage_timeline(score, portfolio, trades, {"txo_point_value": 50.0})

    assert coverage.loc[0, "active_put_spread_max_protection"] == 100_000.0
    assert coverage.loc[0, "current_hedge_coverage"] == 0.10


def test_hedge_gap_audit_does_not_mutate_trades_or_create_trades() -> None:
    coverage = pd.DataFrame(
        {
            "date": pd.date_range("2019-07-01", periods=200, freq="D"),
            "HedgeNeedScore": [60.0] * 200,
            "target_hedge_coverage": [0.35] * 200,
            "current_hedge_coverage": [0.0] * 200,
            "should_attempt_hedge": [True] * 200,
            "if_not_attempted_reason": [""] * 200,
        }
    )
    options = pd.DataFrame()
    audit = hedge_gap_audit(coverage, options, {}, {"target_dte_min": 60, "target_dte_max": 120})

    assert not audit.empty
    assert "crash_window_audit" in set(audit["section"])


def test_hedge_need_report_has_no_best_or_recommend() -> None:
    text = _markdown(pd.DataFrame(), pd.DataFrame()).lower()

    assert "best" not in text
    assert "recommend" not in text


def test_attribution_weighted_sum_equals_score() -> None:
    market = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"]),
            "tx_close": [100.0],
            "ma200": [100.0],
            "ret_126d": [0.0],
            "drawdown_20d_from_high": [0.0],
            "vix_percentile_3y": [20.0],
            "ValuationRiskIndex": [50.0],
            "MacroDemandFragilityIndex": [40.0],
            "SupplyStressIndex": [30.0],
            "LiquidityStressIndex": [20.0],
            "vix_is_proxy": [False],
        }
    )
    score = hedge_need_score_frame(market)
    attribution = hedge_need_score_attribution(score, market)
    daily = attribution[attribution["section"] == "daily_attribution"].iloc[0]
    weighted = sum(float(daily[f"{module}_weighted_contribution"]) for module in RISK_WEIGHTS)

    assert abs(weighted - float(daily["HedgeNeedScore"])) < 1e-9


def test_vix_proxy_confidence_not_high() -> None:
    market = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"]),
            "tx_close": [100.0],
            "ma200": [100.0],
            "ret_126d": [0.0],
            "drawdown_20d_from_high": [0.0],
            "vix_percentile_3y": [20.0],
            "ValuationRiskIndex": [50.0],
            "MacroDemandFragilityIndex": [40.0],
            "SupplyStressIndex": [30.0],
            "LiquidityStressIndex": [20.0],
            "vix_is_proxy": [True],
        }
    )
    score = hedge_need_score_frame(market)
    attribution = hedge_need_score_attribution(score, market)
    confidence = attribution[attribution["section"] == "daily_attribution"]["score_confidence"].iloc[0]

    assert confidence != "HIGH"


def test_attribution_does_not_modify_score_frame() -> None:
    market = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"]),
            "tx_close": [100.0],
            "ma200": [100.0],
            "ret_126d": [0.0],
            "drawdown_20d_from_high": [0.0],
            "vix_percentile_3y": [20.0],
            "ValuationRiskIndex": [50.0],
            "MacroDemandFragilityIndex": [40.0],
            "SupplyStressIndex": [30.0],
            "LiquidityStressIndex": [20.0],
            "vix_is_proxy": [False],
        }
    )
    score = hedge_need_score_frame(market)
    before = score.copy(deep=True)
    hedge_need_score_attribution(score, market)

    pd.testing.assert_frame_equal(score, before)


def test_attribution_report_has_no_best_or_recommend() -> None:
    text = _attribution_markdown(pd.DataFrame()).lower()

    assert "best" not in text
    assert "recommend" not in text


def test_crash_attribution_uses_existing_scores_only() -> None:
    dates = pd.date_range("2019-07-01", periods=200, freq="D")
    market = pd.DataFrame(
        {
            "date": dates,
            "tx_close": [100.0] * len(dates),
            "ma200": [100.0] * len(dates),
            "ret_126d": [0.0] * len(dates),
            "drawdown_20d_from_high": [0.0] * len(dates),
            "vix_percentile_3y": [50.0] * len(dates),
            "ValuationRiskIndex": [50.0] * len(dates),
            "MacroDemandFragilityIndex": [0.0] * len(dates),
            "SupplyStressIndex": [0.0] * len(dates),
            "LiquidityStressIndex": [50.0] * len(dates),
            "vix_is_proxy": [False] * len(dates),
        }
    )
    score = hedge_need_score_frame(market)
    attribution = hedge_need_score_attribution(score, market)
    crash = attribution[(attribution["section"] == "crash_pre_window_attribution") & (attribution["crash_window"] == "2020")]

    assert not crash.empty
    assert float(crash.iloc[0]["score_change"]) == 0.0
