from __future__ import annotations

import pandas as pd

from src.hedge_need import (
    apply_risk_indicators_to_market,
    hedge_coverage_timeline,
    hedge_gap_audit,
    hedge_need_score_frame,
    hedge_need_score_attribution,
    target_hedge_coverage,
    RISK_WEIGHTS,
    _attribution_markdown,
    _markdown,
)
from src.risk_indicator_downloader import read_source_registry, _combine_frames, _audit_markdown
from src.risk_indicator_builder import build_local_risk_indicator_frame, _audit_markdown as _local_audit_markdown


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


def test_risk_source_registry_readable() -> None:
    sources = read_source_registry(__import__("pathlib").Path("data/risk_sources.yml"))

    assert sources
    assert any(source.source_name == "taifex_put_call_ratio" for source in sources)


def test_risk_indicators_missing_columns_do_not_crash() -> None:
    frame = _combine_frames([pd.DataFrame({"date": pd.to_datetime(["2024-01-01"]), "tw_vix": [20.0]})], "2024-01-01", "2024-01-31")

    assert "put_call_volume_ratio" in frame.columns
    assert pd.isna(frame.loc[0, "put_call_volume_ratio"])


def test_tw_vix_available_disables_vix_proxy_for_matching_dates() -> None:
    market = pd.DataFrame({"date": pd.to_datetime(["2024-01-01"]), "vix": [12.0], "vix_is_proxy": [True]})
    risk = pd.DataFrame({"date": pd.to_datetime(["2024-01-01"]), "tw_vix": [22.0]})
    merged = apply_risk_indicators_to_market(market, risk)

    assert merged.loc[0, "vix"] == 22.0
    assert not bool(merged.loc[0, "vix_is_proxy"])


def test_tw_vix_missing_keeps_vix_proxy() -> None:
    market = pd.DataFrame({"date": pd.to_datetime(["2024-01-01"]), "vix": [12.0], "vix_is_proxy": [True]})
    risk = pd.DataFrame({"date": pd.to_datetime(["2024-01-01"]), "tw_vix": [pd.NA]})
    merged = apply_risk_indicators_to_market(market, risk)

    assert merged.loc[0, "vix"] == 12.0
    assert bool(merged.loc[0, "vix_is_proxy"])


def test_risk_indicator_integration_does_not_change_weights() -> None:
    assert RISK_WEIGHTS == {
        "ValuationRisk": 0.20,
        "VolatilityComplacencyRisk": 0.20,
        "MacroDemandFragility": 0.15,
        "SupplyStressIndex": 0.15,
        "LiquidityStress": 0.10,
        "TrendFragility": 0.20,
    }


def test_failed_download_audit_does_not_imply_fill() -> None:
    audit = pd.DataFrame([{"source_name": "x", "status": "failed", "row_count": 0, "detail": "network"}])
    text = _audit_markdown(audit).lower()

    assert "failed" in text
    assert "no mock values" in text


def test_local_put_call_ratio_calculation() -> None:
    market = pd.DataFrame({"date": pd.to_datetime(["2024-01-01"]), "txf_close": [10000.0]})
    options = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"] * 2),
            "expiry": pd.to_datetime(["2024-02-01"] * 2),
            "dte": [30, 30],
            "cp": ["P", "C"],
            "strike": [10000.0, 10000.0],
            "bid": [10.0, 20.0],
            "ask": [12.0, 22.0],
            "volume": [200.0, 100.0],
            "open_interest": [300.0, 100.0],
            "quote_quality_status": ["VALID", "VALID"],
            "is_tradable_quote": [True, True],
        }
    )
    out = build_local_risk_indicator_frame(market, options)

    assert out.loc[0, "put_call_volume_ratio_all"] == 2.0
    assert out.loc[0, "put_call_oi_ratio_all"] == 3.0


def test_local_tradable_only_uses_valid_quote() -> None:
    market = pd.DataFrame({"date": pd.to_datetime(["2024-01-01"]), "txf_close": [10000.0]})
    options = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"] * 3),
            "expiry": pd.to_datetime(["2024-02-01"] * 3),
            "dte": [30, 30, 30],
            "cp": ["P", "P", "C"],
            "strike": [10000.0, 9900.0, 10000.0],
            "bid": [10.0, 99.0, 20.0],
            "ask": [12.0, 101.0, 22.0],
            "volume": [200.0, 1000.0, 100.0],
            "open_interest": [300.0, 1000.0, 100.0],
            "quote_quality_status": ["VALID", "ZERO_BID", "VALID"],
            "is_tradable_quote": [True, False, True],
        }
    )
    out = build_local_risk_indicator_frame(market, options)

    assert out.loc[0, "put_call_volume_ratio_all"] == 12.0
    assert out.loc[0, "put_call_volume_ratio_tradable"] == 2.0


def test_local_atm_selects_nearest_strike() -> None:
    market = pd.DataFrame({"date": pd.to_datetime(["2024-01-01"]), "txf_close": [10050.0]})
    rows = []
    for strike, bid in [(9900.0, 90.0), (10000.0, 10.0), (10200.0, 80.0)]:
        for cp in ["P", "C"]:
            rows.append(
                {
                    "date": pd.Timestamp("2024-01-01"),
                    "expiry": pd.Timestamp("2024-02-01"),
                    "dte": 30,
                    "cp": cp,
                    "strike": strike,
                    "bid": bid,
                    "ask": bid + 2.0,
                    "volume": 100.0,
                    "open_interest": 100.0,
                    "quote_quality_status": "VALID",
                    "is_tradable_quote": True,
                }
            )
    out = build_local_risk_indicator_frame(market, pd.DataFrame(rows))

    assert out.loc[0, "atm_put_premium_ratio_30d"] == (11.0 / 10050.0)


def test_local_missing_data_outputs_nan_not_zero() -> None:
    market = pd.DataFrame({"date": pd.to_datetime(["2024-01-01"]), "txf_close": [10000.0]})
    options = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"]),
            "expiry": pd.to_datetime(["2024-04-01"]),
            "dte": [90],
            "cp": ["P"],
            "strike": [9000.0],
            "bid": [10.0],
            "ask": [12.0],
            "volume": [100.0],
            "open_interest": [100.0],
            "quote_quality_status": ["VALID"],
            "is_tradable_quote": [True],
        }
    )
    out = build_local_risk_indicator_frame(market, options)

    assert pd.isna(out.loc[0, "atm_straddle_premium_ratio_30d"])


def test_local_proxy_confidence_not_high() -> None:
    market = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"] * 60),
            "tx_close": [100.0] * 60,
            "ma200": [100.0] * 60,
            "ret_126d": [0.0] * 60,
            "drawdown_20d_from_high": [0.0] * 60,
            "vix_percentile_3y": [50.0] * 60,
            "ValuationRiskIndex": [50.0] * 60,
            "MacroDemandFragilityIndex": [0.0] * 60,
            "SupplyStressIndex": [0.0] * 60,
            "LiquidityStressIndex": [50.0] * 60,
            "vix_is_proxy": [False] * 60,
            "iv_proxy_source": ["local_txo_chain_proxy"] * 60,
            "atm_straddle_premium_ratio_30d": [0.05] * 60,
        }
    )
    score = hedge_need_score_frame(market)
    attribution = hedge_need_score_attribution(score, market)
    confidence = set(attribution[attribution["section"] == "daily_attribution"]["score_confidence"])

    assert "HIGH" not in confidence


def test_local_proxy_source_type_not_official_vix() -> None:
    market = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"]),
            "tx_close": [100.0],
            "ma200": [100.0],
            "ret_126d": [0.0],
            "drawdown_20d_from_high": [0.0],
            "vix": [20.0],
            "vix_percentile_3y": [50.0],
            "ValuationRiskIndex": [55.0],
            "MacroDemandFragilityIndex": [10.0],
            "SupplyStressIndex": [10.0],
            "LiquidityStressIndex": [55.0],
            "vix_is_proxy": [False],
            "iv_proxy_source": ["local_txo_chain_proxy"],
            "atm_straddle_premium_ratio_30d": [0.05],
        }
    )
    score = hedge_need_score_frame(market)

    assert score.loc[0, "volatility_source_type"] == "LOCAL_TXO_PROXY"


def test_realized_vol_proxy_confidence_low() -> None:
    market = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"]),
            "tx_close": [100.0],
            "ma200": [100.0],
            "ret_126d": [0.0],
            "drawdown_20d_from_high": [0.0],
            "vix": [20.0],
            "vix_percentile_3y": [50.0],
            "ValuationRiskIndex": [55.0],
            "MacroDemandFragilityIndex": [10.0],
            "SupplyStressIndex": [10.0],
            "LiquidityStressIndex": [55.0],
            "vix_is_proxy": [True],
        }
    )
    score = hedge_need_score_frame(market)
    attribution = hedge_need_score_attribution(score, market)

    assert attribution[attribution["section"] == "daily_attribution"]["score_confidence"].iloc[0] == "LOW"


def test_missing_volatility_source_not_high() -> None:
    market = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"]),
            "tx_close": [100.0],
            "ma200": [100.0],
            "ret_126d": [0.0],
            "drawdown_20d_from_high": [0.0],
            "vix": [pd.NA],
            "vix_percentile_3y": [pd.NA],
            "ValuationRiskIndex": [55.0],
            "MacroDemandFragilityIndex": [10.0],
            "SupplyStressIndex": [10.0],
            "LiquidityStressIndex": [55.0],
            "vix_is_proxy": [False],
        }
    )
    score = hedge_need_score_frame(market)
    attribution = hedge_need_score_attribution(score, market)

    assert attribution[attribution["section"] == "daily_attribution"]["score_confidence"].iloc[0] != "HIGH"


def test_default_flat_macro_quality_not_high() -> None:
    market = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"]),
            "tx_close": [100.0],
            "ma200": [100.0],
            "ret_126d": [0.0],
            "drawdown_20d_from_high": [0.0],
            "vix": [20.0],
            "vix_percentile_3y": [50.0],
            "ValuationRiskIndex": [50.0],
            "MacroDemandFragilityIndex": [0.0],
            "SupplyStressIndex": [0.0],
            "LiquidityStressIndex": [50.0],
            "vix_is_proxy": [False],
        }
    )
    score = hedge_need_score_frame(market)
    attribution = hedge_need_score_attribution(score, market)

    assert attribution[attribution["section"] == "daily_attribution"]["score_confidence"].iloc[0] != "HIGH"


def test_local_risk_indicator_report_has_no_best_or_recommend() -> None:
    text = _local_audit_markdown(pd.DataFrame()).lower()

    assert "best" not in text
    assert "recommend" not in text
