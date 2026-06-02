from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.risk_scaled_hedge import (
    RiskScaledHedgeStateMachine,
    _markdown,
    calculate_hedge_gap,
    run_risk_scaled_hedge_simulation,
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
        "risk_scaled_hedge_gap_threshold": 0.03,
        "risk_scaled_max_contracts_per_day": 10,
        "risk_scaled_max_total_hedge_coverage": 0.80,
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


def _market(dates: list[str], target: float = 0.30, score: float = 60.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "tx_close": [10000.0] * len(dates),
            "txf_close": [10000.0] * len(dates),
            "ma200": [9000.0] * len(dates),
            "ret_126d": [0.0] * len(dates),
            "vix_percentile_3y": [50.0] * len(dates),
            "event_flag": [0] * len(dates),
            "vix": [20.0] * len(dates),
            "vix_5ma": [20.0] * len(dates),
            "macro_state": ["NORMAL"] * len(dates),
            "SupplyStressIndex": [0.0] * len(dates),
            "MacroDemandFragilityIndex": [0.0] * len(dates),
            "CombinedRiskScore": [0.0] * len(dates),
            "ValuationRiskIndex": [50.0] * len(dates),
            "HedgeNeedScore": [score] * len(dates),
            "target_hedge_coverage": [target] * len(dates),
            "score_confidence": ["LOW"] * len(dates),
            "volatility_source_type": ["LOCAL_TXO_PROXY"] * len(dates),
        }
    )


def _portfolio(dates: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"date": pd.to_datetime(dates), "stock_equity": [1_200_000.0] * len(dates), "portfolio_beta": [1.3] * len(dates)})


def _options(date: str, valid: bool = True, expired: bool = False) -> pd.DataFrame:
    expiry = str((pd.Timestamp(date) + pd.Timedelta(days=106)).date()) if not expired else "2023-12-20"
    dte = 106 if not expired else -13
    return pd.DataFrame(
        {
            "date": pd.to_datetime([date, date]),
            "expiry": pd.to_datetime([expiry, expiry]),
            "dte": [dte, dte],
            "cp": ["P", "P"],
            "strike": [9000.0, 7500.0],
            "close": [20.0, 10.0],
            "bid": [19.0, 9.0],
            "ask": [21.0, 11.0],
            "volume": [500.0, 500.0],
            "open_interest": [1000.0, 1000.0],
            "iv": [0.2, 0.2],
            "delta": [-0.1, -0.05],
            "underlying": [10000.0, 10000.0],
            "tradable": [valid, valid],
            "reason": ["", ""],
            "bid_ask_estimated": [False, False],
            "iv_estimated": [False, False],
            "delta_estimated": [False, False],
            "quote_quality_status": ["VALID" if valid else "ZERO_BID"] * 2,
            "spread_pct": [0.1, 0.1],
            "is_tradable_quote": [valid, valid],
        }
    )


def _run_engine(dates: list[str], target: float = 0.30, valid: bool = True, config: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = _config() if config is None else config
    options = pd.concat([_options(date, valid=valid) for date in dates], ignore_index=True)
    engine = RiskScaledHedgeStateMachine(
        _market(dates, target=target),
        options,
        _portfolio(dates),
        cfg,
        _put_params(),
        _ic_params(),
        mode="put_spread_only",
    )
    equity, trades = engine.run()
    return equity, trades, pd.DataFrame(engine.coverage_rows)


def test_hedge_gap_calculation_is_target_minus_current() -> None:
    assert calculate_hedge_gap(0.35, 0.12) == pytest.approx(0.23)


def test_gap_below_threshold_does_not_enter() -> None:
    _, trades, coverage = _run_engine(["2024-01-02"], target=0.02)

    assert trades.empty
    assert coverage.iloc[0]["rejection_reason"] == "HEDGE_GAP_TOO_SMALL"


def test_all_trades_use_valid_quote() -> None:
    _, trades, _ = _run_engine(["2024-01-02"], target=0.30)

    assert not trades.empty
    assert set(trades["quote_quality_status"]) == {"VALID"}
    assert trades["is_tradable_quote"].all()


def test_non_valid_quote_is_blocked() -> None:
    _, trades, coverage = _run_engine(["2024-01-02"], target=0.30, valid=False)

    assert trades.empty
    assert coverage.iloc[0]["rejection_reason"] in {"QUOTE_NOT_VALID", "LOW_LIQUIDITY", "NO_CONTRACT_FOUND", "UNKNOWN"}


def test_annual_budget_cannot_be_breached() -> None:
    cfg = _config()
    cfg["max_annual_hedge_budget_pct"] = 0.001
    _, trades, coverage = _run_engine(["2024-01-02", "2024-01-03"], target=0.80, config=cfg)

    assert not trades.empty
    assert pd.to_numeric(coverage["annual_budget_used"], errors="coerce").max() <= 1_200_000.0 * 0.001 + 1e-8


def test_no_lookahead_current_day_options_only() -> None:
    dates = ["2024-01-02"]
    engine = RiskScaledHedgeStateMachine(
        _market(dates, target=0.30),
        _options("2024-01-03"),
        _portfolio(dates),
        _config(),
        _put_params(),
        _ic_params(),
        mode="put_spread_only",
    )
    _, trades = engine.run()

    assert trades.empty
    assert pd.DataFrame(engine.coverage_rows).iloc[0]["rejection_reason"] == "NO_CONTRACT_FOUND"


def test_simulation_does_not_modify_hedge_need_score_file(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    report_dir = tmp_path / "reports"
    data_dir.mkdir()
    report_dir.mkdir()
    (report_dir / "hedge_need_score.csv").write_text("date,HedgeNeedScore\n2024-01-02,50\n", encoding="utf-8")
    before = (report_dir / "hedge_need_score.csv").read_text(encoding="utf-8")
    market = _market(["2024-01-02"], target=0.30)
    market[["date", "tx_close", "txf_close", "vix", "event_flag"]].to_csv(data_dir / "market.csv", index=False)
    _options("2024-01-02").drop(columns=["underlying"]).to_csv(data_dir / "options.csv", index=False)

    run_risk_scaled_hedge_simulation(data_dir, report_dir, _config(), _put_params(), _ic_params())

    assert (report_dir / "hedge_need_score.csv").read_text(encoding="utf-8") == before


def test_target_coverage_is_score_driven_not_crash_window_hardcoded(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    report_dir = tmp_path / "reports"
    data_dir.mkdir()
    report_dir.mkdir()
    market = _market(["2020-03-10"], target=0.80)
    market[["date", "tx_close", "txf_close", "vix", "event_flag"]].to_csv(data_dir / "market.csv", index=False)
    _options("2020-03-10").drop(columns=["underlying"]).to_csv(data_dir / "options.csv", index=False)

    _, _, coverage = run_risk_scaled_hedge_simulation(data_dir, report_dir, _config(), _put_params(), _ic_params())

    assert not coverage.empty
    assert coverage.iloc[0]["date"] == "2020-03-10"
    assert coverage.iloc[0]["target_hedge_coverage"] < 0.50


def test_report_has_no_disallowed_wording() -> None:
    summary = pd.DataFrame([{"section": "summary", "metric": "average_target_coverage", "value": 0.1}])
    text = _markdown(summary, pd.DataFrame(), pd.DataFrame()).lower()

    assert "best" not in text
    assert "recommended" not in text
