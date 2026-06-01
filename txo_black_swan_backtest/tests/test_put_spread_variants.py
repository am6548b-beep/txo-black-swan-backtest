from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.put_spread_variants import (
    PutSpreadVariantStateMachine,
    _coverage_timeline_rows,
    _entry_time_feasibility_filter,
    _exit_timing_diagnostic,
    _gap_summary_rows,
    _leg_quote,
    _rolling_coverage_markdown,
    _run_exit_timing_variants,
    _run_moneyness_tradability_diagnostic,
    run_put_spread_variants,
)


def _config() -> dict:
    return {
        "initial_cash": 300_000.0,
        "initial_stock_equity": 1_200_000.0,
        "portfolio_beta": 1.3,
        "txo_point_value": 50.0,
        "commission_per_contract_per_side": 30.0,
        "option_tax_rate_on_premium": 0.001,
        "normal_slippage_pct": 0.03,
        "stress_slippage_pct": 0.12,
        "wide_spread_volume_threshold": 50,
        "min_open_interest": 100,
        "max_annual_hedge_budget_pct": 0.03,
        "base_annual_hedge_budget_pct": 0.015,
        "max_margin_usage_pct": 0.35,
        "min_free_cash_multiplier": 2.0,
        "risk_free_rate": 0.015,
        "debug_mid_fill": False,
        "stress_vix_percentile": 80.0,
    }


def _put_params() -> dict:
    return {
        "long_put_moneyness": 0.90,
        "long_put_moneyness_low_vix": 0.93,
        "short_put_moneyness": 0.75,
        "short_put_moneyness_low_vix": 0.78,
        "target_dte_min": 60,
        "target_dte_max": 120,
        "profit_take_1": 0.60,
        "profit_take_2": 0.80,
        "drop_take_1": -0.15,
        "drop_take_2": -0.20,
        "exit_dte": 14,
        "low_value_exit_pct": 0.10,
        "low_value_exit_dte": 30,
    }


def _ic_params() -> dict:
    return {"target_dte_min": 30, "target_dte_max": 45}


def _market(dates: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "tx_close": [10000.0] * len(dates),
            "txf_close": [10000.0] * len(dates),
            "ma200": [20000.0] * len(dates),
            "ret_126d": [-0.50] * len(dates),
            "vix_percentile_3y": [90.0] * len(dates),
            "event_flag": [0] * len(dates),
            "vix": [20.0] * len(dates),
            "vix_5ma": [20.0] * len(dates),
            "macro_state": ["NORMAL"] * len(dates),
            "SupplyStressIndex": [0.0] * len(dates),
            "MacroDemandFragilityIndex": [0.0] * len(dates),
            "CombinedRiskScore": [0.0] * len(dates),
            "ValuationRiskIndex": [0.0] * len(dates),
        }
    )


def _portfolio(dates: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"date": pd.to_datetime(dates), "stock_equity": [1_200_000.0] * len(dates), "portfolio_beta": [1.3] * len(dates)})


def _options(date: str, valid: bool = True, expired: bool = False) -> pd.DataFrame:
    expiry = str((pd.Timestamp(date) + pd.Timedelta(days=106)).date()) if not expired else "2023-12-20"
    dte = 106 if not expired else -13
    return pd.DataFrame(
        {
            "date": pd.to_datetime([date, date, date]),
            "expiry": pd.to_datetime([expiry, expiry, expiry]),
            "dte": [dte, dte, dte],
            "cp": ["P", "P", "P"],
            "strike": [9000.0, 7500.0, 7400.0],
            "close": [20.0, 10.0, 8.0],
            "bid": [19.0, 9.0, 7.0],
            "ask": [21.0, 11.0, 9.0],
            "volume": [500.0, 500.0, 500.0],
            "open_interest": [1000.0, 1000.0, 1000.0],
            "iv": [0.2, 0.2, 0.2],
            "delta": [-0.1, -0.05, -0.04],
            "underlying": [10000.0, 10000.0, 10000.0],
            "tradable": [valid, valid, valid],
            "reason": ["", "", ""],
            "bid_ask_estimated": [False, False, False],
            "iv_estimated": [False, False, False],
            "delta_estimated": [False, False, False],
            "quote_quality_status": ["VALID" if valid else "ZERO_BID"] * 3,
            "spread_pct": [0.1, 0.1, 0.1],
            "is_tradable_quote": [valid, valid, valid],
        }
    )


def _history_options(entry_date: str, past_valid_days: int, past_invalid_days: int, future_valid_days: int = 0) -> pd.DataFrame:
    entry = pd.Timestamp(entry_date)
    expiry = str((entry + pd.Timedelta(days=106)).date())
    rows = []
    for i in range(past_invalid_days):
        rows.append((entry - pd.Timedelta(days=past_invalid_days + past_valid_days - i), "ZERO_BID", False, 0, 0, 1.2))
    for i in range(past_valid_days):
        rows.append((entry - pd.Timedelta(days=past_valid_days - i), "VALID", True, 500, 1000, 0.1))
    rows.append((entry, "VALID", True, 500, 1000, 0.1))
    for i in range(future_valid_days):
        rows.append((entry + pd.Timedelta(days=i + 1), "VALID", True, 500, 1000, 0.1))
    out = []
    for date, status, tradable, volume, oi, spread in rows:
        for strike in [9000.0, 7500.0]:
            out.append(
                {
                    "date": date,
                    "expiry": pd.Timestamp(expiry),
                    "dte": 106,
                    "cp": "P",
                    "strike": strike,
                    "close": 20.0,
                    "bid": 19.0,
                    "ask": 21.0,
                    "volume": volume,
                    "open_interest": oi,
                    "iv": 0.2,
                    "delta": -0.1,
                    "underlying": 10000.0,
                    "tradable": tradable,
                    "reason": "",
                    "bid_ask_estimated": False,
                    "iv_estimated": False,
                    "delta_estimated": False,
                    "quote_quality_status": status,
                    "spread_pct": spread,
                    "is_tradable_quote": tradable,
                }
            )
    return pd.DataFrame(out)


def test_quarterly_base_insurance_does_not_depend_on_original_market_signal() -> None:
    dates = ["2024-01-02"]
    engine = PutSpreadVariantStateMachine(
        _market(dates),
        _options("2024-01-02"),
        _portfolio(dates),
        _config(),
        _put_params(),
        _ic_params(),
        mode="put_spread_only",
        variant_name="quarterly_base_insurance",
    )
    _, trades = engine.run()
    assert len(trades) == 2
    assert set(trades["quote_quality_status"]) == {"VALID"}


def test_base_plus_signal_boost_does_not_exceed_annual_budget() -> None:
    dates = ["2024-01-02", "2024-04-01", "2024-07-01", "2024-10-01"]
    options = pd.concat([_options(date) for date in dates], ignore_index=True)
    engine = PutSpreadVariantStateMachine(
        _market(dates),
        options,
        _portfolio(dates),
        _config(),
        _put_params(),
        _ic_params(),
        mode="put_spread_only",
        variant_name="base_plus_signal_boost",
    )
    _, trades = engine.run()
    opens = trades[trades["reason"].astype(str).str.contains("quarterly")]
    annual_cost = -opens["cash_flow"].sum()
    assert annual_cost <= _config()["max_annual_hedge_budget_pct"] * _config()["initial_stock_equity"]


def test_variants_do_not_trade_non_valid_quotes() -> None:
    dates = ["2024-01-02"]
    engine = PutSpreadVariantStateMachine(
        _market(dates),
        _options("2024-01-02", valid=False),
        _portfolio(dates),
        _config(),
        _put_params(),
        _ic_params(),
        mode="put_spread_only",
        variant_name="quarterly_base_insurance",
    )
    _, trades = engine.run()
    assert trades.empty


def test_variants_do_not_trade_expired_options() -> None:
    dates = ["2024-01-02"]
    engine = PutSpreadVariantStateMachine(
        _market(dates),
        _options("2024-01-02", expired=True),
        _portfolio(dates),
        _config(),
        _put_params(),
        _ic_params(),
        mode="put_spread_only",
        variant_name="quarterly_base_insurance",
    )
    _, trades = engine.run()
    assert trades.empty


def test_variant_comparison_does_not_select_best_parameter(tmp_path: Path) -> None:
    dates = ["2024-01-02"]
    data_dir = tmp_path / "data"
    report_dir = tmp_path / "reports"
    data_dir.mkdir()
    _market(dates).assign(tx_open=10000.0, tx_high=10000.0, tx_low=10000.0, volume=1, vix_is_proxy=False)[
        ["date", "tx_close", "tx_open", "tx_high", "tx_low", "txf_close", "volume", "vix", "event_flag"]
    ].to_csv(data_dir / "market.csv", index=False)
    _options("2024-01-02").drop(columns=["underlying", "tradable", "reason"]).to_csv(data_dir / "options.csv", index=False)
    _portfolio(dates).to_csv(data_dir / "portfolio.csv", index=False)

    comparison = run_put_spread_variants(data_dir, report_dir, _config(), _put_params(), _ic_params())
    assert "best" not in " ".join(comparison.columns).lower()
    assert "best" not in (report_dir / "put_spread_variant_comparison.md").read_text(encoding="utf-8").lower()


def test_quarterly_entry_uses_current_quarter_check_without_future_retry() -> None:
    dates = ["2024-01-02", "2024-01-03"]
    engine = PutSpreadVariantStateMachine(
        _market(dates),
        _options("2024-01-03"),
        _portfolio(dates),
        _config(),
        _put_params(),
        _ic_params(),
        mode="put_spread_only",
        variant_name="quarterly_base_insurance",
    )
    _, trades = engine.run()
    assert trades.empty


def test_variant_d_does_not_modify_quarterly_base_result() -> None:
    dates = ["2024-01-02"]
    options = _history_options("2024-01-02", past_valid_days=0, past_invalid_days=10)
    quarterly = PutSpreadVariantStateMachine(
        _market(dates),
        options,
        _portfolio(dates),
        _config(),
        _put_params(),
        _ic_params(),
        mode="put_spread_only",
        variant_name="quarterly_base_insurance",
    )
    variant_d = PutSpreadVariantStateMachine(
        _market(dates),
        options,
        _portfolio(dates),
        _config(),
        _put_params(),
        _ic_params(),
        mode="put_spread_only",
        variant_name="quarterly_base_insurance_feasible_only",
    )

    _, quarterly_trades = quarterly.run()
    _, d_trades = variant_d.run()

    assert len(quarterly_trades) == 2
    assert d_trades.empty


def test_variant_d_excludes_only_exit_unlikely_not_fragile() -> None:
    dates = ["2024-01-02"]
    options = _history_options("2024-01-02", past_valid_days=1, past_invalid_days=0)
    engine = PutSpreadVariantStateMachine(
        _market(dates),
        options,
        _portfolio(dates),
        _config(),
        _put_params(),
        _ic_params(),
        mode="put_spread_only",
        variant_name="quarterly_base_insurance_feasible_only",
    )
    _, trades = engine.run()
    assert len(trades) == 2


def test_entry_time_feasibility_filter_ignores_future_data() -> None:
    options = _history_options("2024-01-02", past_valid_days=0, past_invalid_days=10, future_valid_days=20)
    market = _market(["2024-01-02"])
    long = _options("2024-01-02").iloc[0]
    short = _options("2024-01-02").iloc[1]
    from src.strategies import row_to_contract

    long_contract = row_to_contract(long)
    short_contract = row_to_contract(short)
    result = _entry_time_feasibility_filter(long_contract, short_contract, options, _config(), _put_params())

    assert result["filter_decision"] == "SKIP_EXIT_UNLIKELY"
    assert result["no_lookahead_pass"] is True
    assert pd.Timestamp(result["long_filter_max_reference_date"]) <= pd.Timestamp("2024-01-02")


def test_exit_timing_diagnostic_does_not_modify_trades() -> None:
    dates = ["2024-01-02"]
    options = _history_options("2024-01-02", past_valid_days=3, past_invalid_days=0)
    engine = PutSpreadVariantStateMachine(
        _market(dates),
        options,
        _portfolio(dates),
        _config(),
        _put_params(),
        _ic_params(),
        mode="put_spread_only",
        variant_name="quarterly_base_insurance",
    )
    equity, trades = engine.run()
    before = trades.copy(deep=True)
    _exit_timing_diagnostic(
        [{"variant": "quarterly_base_insurance", "trades": trades, "lifecycle": pd.DataFrame(equity.attrs.get("position_lifecycle_events", []))}],
        options,
        _config(),
        _put_params(),
    )
    pd.testing.assert_frame_equal(before, trades)


def test_last_valid_full_spread_requires_both_legs_valid() -> None:
    options = _history_options("2024-01-02", past_valid_days=0, past_invalid_days=0)
    expiry = pd.Timestamp("2024-04-17")
    date = expiry - pd.Timedelta(days=30)
    rows = []
    rows.append({**_options("2024-01-02").iloc[0].to_dict(), "date": date, "expiry": expiry, "quote_quality_status": "VALID", "is_tradable_quote": True, "volume": 500, "open_interest": 1000, "strike": 9000.0})
    rows.append({**_options("2024-01-02").iloc[1].to_dict(), "date": date, "expiry": expiry, "quote_quality_status": "ZERO_BID", "is_tradable_quote": False, "volume": 500, "open_interest": 1000, "strike": 7500.0})
    options = pd.DataFrame(rows)
    trades = pd.DataFrame(
        [
            {"date": "2024-01-02", "position_id": "PS-1", "strategy": "put_spread", "action": "BUY", "cp": "P", "strike": 9000.0, "expiry": "2024-04-17", "quantity": 1, "cash_flow": -1000, "reason": "quarterly_base_put_spread", "dte_at_trade": 106, "txf_close_at_trade": 10000},
            {"date": "2024-01-02", "position_id": "PS-1", "strategy": "put_spread", "action": "SELL", "cp": "P", "strike": 7500.0, "expiry": "2024-04-17", "quantity": -1, "cash_flow": 100, "reason": "quarterly_base_put_spread", "dte_at_trade": 106, "txf_close_at_trade": 10000},
        ]
    )
    out = _exit_timing_diagnostic([{"variant": "quarterly_base_insurance", "trades": trades, "lifecycle": pd.DataFrame()}], options, _config(), _put_params())
    pos = out[out["section"] == "position_exit_timing"].iloc[0]
    assert pd.isna(pos["last_day_both_legs_valid"]) or pos["last_day_both_legs_valid"] == ""


def test_hypothetical_exit_availability_does_not_create_trades() -> None:
    dates = ["2024-01-02"]
    options = _history_options("2024-01-02", past_valid_days=3, past_invalid_days=0)
    engine = PutSpreadVariantStateMachine(
        _market(dates),
        options,
        _portfolio(dates),
        _config(),
        _put_params(),
        _ic_params(),
        mode="put_spread_only",
        variant_name="quarterly_base_insurance",
    )
    equity, trades = engine.run()
    count_before = len(trades)
    _exit_timing_diagnostic(
        [{"variant": "quarterly_base_insurance", "trades": trades, "lifecycle": pd.DataFrame(equity.attrs.get("position_lifecycle_events", []))}],
        options,
        _config(),
        _put_params(),
    )
    assert len(trades) == count_before


def test_exit_timing_markdown_does_not_recommend_best_parameter(tmp_path: Path) -> None:
    dates = ["2024-01-02"]
    data_dir = tmp_path / "data"
    report_dir = tmp_path / "reports"
    data_dir.mkdir()
    _market(dates).assign(tx_open=10000.0, tx_high=10000.0, tx_low=10000.0, volume=1)[
        ["date", "tx_close", "tx_open", "tx_high", "tx_low", "txf_close", "volume", "vix", "event_flag"]
    ].to_csv(data_dir / "market.csv", index=False)
    _options("2024-01-02").drop(columns=["underlying", "tradable", "reason"]).to_csv(data_dir / "options.csv", index=False)
    _portfolio(dates).to_csv(data_dir / "portfolio.csv", index=False)
    run_put_spread_variants(data_dir, report_dir, _config(), _put_params(), _ic_params())
    text = (report_dir / "exit_timing_diagnostic.md").read_text(encoding="utf-8").lower()
    assert "best dte" not in text
    assert "recommend" not in text


def test_exit_timing_variants_keep_quarterly_entry_rules() -> None:
    dates = ["2024-01-02"]
    market = _market(dates)
    options = _history_options("2024-01-02", past_valid_days=10, past_invalid_days=0)
    comparison = _run_exit_timing_variants(market, options, _portfolio(dates), _config(), _put_params(), _ic_params())
    summary = comparison[comparison["section"] == "exit_variant_summary"]

    assert set(summary["position_count"]) == {1}
    assert set(summary["variant"]) == {
        "quarterly_base_insurance_exit_dte30",
        "quarterly_base_insurance_exit_dte21",
        "quarterly_base_insurance_exit_dte14",
    }


def test_exit_timing_variants_do_not_trade_non_valid_or_after_expiry() -> None:
    dates = ["2024-01-02"]
    market = _market(dates)
    options = _history_options("2024-01-02", past_valid_days=10, past_invalid_days=0)
    comparison = _run_exit_timing_variants(market, options, _portfolio(dates), _config(), _put_params(), _ic_params())
    audit = comparison[comparison["section"] == "exit_variant_audit"]

    assert (audit[audit["check"] == "non_valid_quote_trades_count"]["count"] == 0).all()
    assert (audit[audit["check"] == "trade_date_after_expiry_count"]["count"] == 0).all()


def test_exit_timing_variant_report_does_not_rank_or_recommend(tmp_path: Path) -> None:
    dates = ["2024-01-02"]
    data_dir = tmp_path / "data"
    report_dir = tmp_path / "reports"
    data_dir.mkdir()
    _market(dates).assign(tx_open=10000.0, tx_high=10000.0, tx_low=10000.0, volume=1)[
        ["date", "tx_close", "tx_open", "tx_high", "tx_low", "txf_close", "volume", "vix", "event_flag"]
    ].to_csv(data_dir / "market.csv", index=False)
    _options("2024-01-02").drop(columns=["underlying", "tradable", "reason"]).to_csv(data_dir / "options.csv", index=False)
    _portfolio(dates).to_csv(data_dir / "portfolio.csv", index=False)

    run_put_spread_variants(data_dir, report_dir, _config(), _put_params(), _ic_params())
    text = (report_dir / "exit_timing_variant_comparison.md").read_text(encoding="utf-8").lower()

    assert "best" not in text
    assert "recommend" not in text


def test_original_quarterly_variant_not_overwritten_by_exit_timing_variants(tmp_path: Path) -> None:
    dates = ["2024-01-02"]
    data_dir = tmp_path / "data"
    report_dir = tmp_path / "reports"
    data_dir.mkdir()
    _market(dates).assign(tx_open=10000.0, tx_high=10000.0, tx_low=10000.0, volume=1)[
        ["date", "tx_close", "tx_open", "tx_high", "tx_low", "txf_close", "volume", "vix", "event_flag"]
    ].to_csv(data_dir / "market.csv", index=False)
    _options("2024-01-02").drop(columns=["underlying", "tradable", "reason"]).to_csv(data_dir / "options.csv", index=False)
    _portfolio(dates).to_csv(data_dir / "portfolio.csv", index=False)

    run_put_spread_variants(data_dir, report_dir, _config(), _put_params(), _ic_params())
    base = pd.read_csv(report_dir / "put_spread_variant_comparison.csv")
    timing = pd.read_csv(report_dir / "exit_timing_variant_comparison.csv")

    assert "quarterly_base_insurance" in set(base["variant"].dropna())
    assert "quarterly_base_insurance_exit_dte14" in set(timing["variant"].dropna())


def test_moneyness_diagnostic_does_not_modify_original_params() -> None:
    dates = ["2024-01-02"]
    params = _put_params()
    before = params.copy()
    _run_moneyness_tradability_diagnostic(_market(dates), _options("2024-01-02"), _portfolio(dates), _config(), params, _ic_params())
    assert params == before


def test_moneyness_diagnostic_uses_valid_quotes_and_no_expired_trades() -> None:
    dates = ["2024-01-02"]
    diagnostic = _run_moneyness_tradability_diagnostic(_market(dates), _options("2024-01-02"), _portfolio(dates), _config(), _put_params(), _ic_params())
    audit = diagnostic[diagnostic["section"] == "moneyness_audit"]

    assert (audit[audit["check"] == "non_valid_quote_trades_count"]["count"] == 0).all()
    assert (audit[audit["check"] == "trade_date_after_expiry_count"]["count"] == 0).all()


def test_moneyness_report_does_not_rank_or_recommend(tmp_path: Path) -> None:
    dates = ["2024-01-02"]
    data_dir = tmp_path / "data"
    report_dir = tmp_path / "reports"
    data_dir.mkdir()
    _market(dates).assign(tx_open=10000.0, tx_high=10000.0, tx_low=10000.0, volume=1)[
        ["date", "tx_close", "tx_open", "tx_high", "tx_low", "txf_close", "volume", "vix", "event_flag"]
    ].to_csv(data_dir / "market.csv", index=False)
    _options("2024-01-02").drop(columns=["underlying", "tradable", "reason"]).to_csv(data_dir / "options.csv", index=False)
    _portfolio(dates).to_csv(data_dir / "portfolio.csv", index=False)

    run_put_spread_variants(data_dir, report_dir, _config(), _put_params(), _ic_params())
    text = (report_dir / "moneyness_tradability_diagnostic.md").read_text(encoding="utf-8").lower()

    assert "best" not in text
    assert "recommend" not in text


def test_rolling_gap_calculation_is_correct() -> None:
    market = _market(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    trades = pd.DataFrame(
        [
            {"date": "2024-01-02", "position_id": "PS-1", "expiry": "2024-04-17", "reason": "quarterly_base_put_spread"},
            {"date": "2024-01-02", "position_id": "PS-1", "expiry": "2024-04-17", "reason": "quarterly_base_put_spread"},
            {"date": "2024-01-03", "position_id": "PS-1", "expiry": "2024-04-17", "reason": "dte_exit"},
            {"date": "2024-01-03", "position_id": "PS-1", "expiry": "2024-04-17", "reason": "dte_exit"},
        ]
    )
    timeline = _coverage_timeline_rows("quarterly_base_insurance", trades, pd.DataFrame(), market)
    summary = _gap_summary_rows("quarterly_base_insurance", timeline)[0]

    assert int(summary["covered_days"]) == 2
    assert int(summary["uncovered_days"]) == 3
    assert int(summary["longest_uncovered_gap_days"]) == 2


def test_rolling_gap_analysis_does_not_modify_trades() -> None:
    market = _market(["2024-01-01", "2024-01-02", "2024-01-03"])
    trades = pd.DataFrame(
        [
            {"date": "2024-01-02", "position_id": "PS-1", "expiry": "2024-04-17", "reason": "quarterly_base_put_spread"},
            {"date": "2024-01-02", "position_id": "PS-1", "expiry": "2024-04-17", "reason": "quarterly_base_put_spread"},
        ]
    )
    before = trades.copy(deep=True)
    _coverage_timeline_rows("quarterly_base_insurance", trades, pd.DataFrame(), market)
    pd.testing.assert_frame_equal(trades, before)


def test_rolling_crash_prewindow_does_not_generate_trades_from_future() -> None:
    market = _market(["2024-01-01", "2024-01-02"])
    trades = pd.DataFrame(columns=["date", "position_id", "expiry", "reason"])
    timeline = _coverage_timeline_rows("quarterly_base_insurance", trades, pd.DataFrame(), market)

    assert timeline["has_active_put_spread"].sum() == 0
    assert trades.empty


def test_rolling_coverage_report_does_not_rank_or_recommend() -> None:
    diagnostic = pd.DataFrame(
        [
            {
                "section": "gap_summary",
                "variant": "quarterly_base_insurance",
                "coverage_ratio": 0.1,
                "longest_uncovered_gap_days": 10,
                "number_of_gaps_gt_180d": 0,
            }
        ]
    )
    text = _rolling_coverage_markdown(diagnostic).lower()

    assert "best" not in text
    assert "recommend" not in text
