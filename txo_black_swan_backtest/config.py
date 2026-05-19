"""Default configuration for the TXO black-swan hedge backtester.

The values here are deliberately coarse and explainable. They are not the
result of a performance optimization pass.
"""

from __future__ import annotations

BASE_CONFIG = {
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
    "estimated_spread": 0.08,
    "max_annual_hedge_budget_pct": 0.03,
    "base_annual_hedge_budget_pct": 0.015,
    "max_margin_usage_pct": 0.35,
    "min_free_cash_multiplier": 2.0,
    "risk_free_rate": 0.015,
    "debug_mid_fill": False,
    "stress_vix_percentile": 80.0,
}

PUT_SPREAD_PARAMS = {
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

IRON_CONDOR_PARAMS = {
    "target_dte_min": 30,
    "target_dte_max": 45,
    "short_put_delta": -0.05,
    "short_call_delta": 0.15,
    "wing_width": 200,
    "batch_weights": (0.30, 0.30, 0.40),
    "profit_take_pct": 0.50,
    "high_stress_profit_take_pct": 0.40,
    "reduce_delta": 0.30,
    "stop_delta": 0.45,
    "exit_dte": 10,
    "max_total_asset_loss_pct": 0.015,
    "max_hedge_profit_giveback_pct": 0.40,
}

COARSE_PARAM_GRID = {
    "long_put_moneyness": [0.90, 0.93],
    "short_put_moneyness": [0.75, 0.78],
    "put_spread_dte": [(60, 90), (90, 120)],
    "iron_condor_put_delta": [0.04, 0.05, 0.06],
    "iron_condor_call_delta": [0.12, 0.15],
    "wing_width": [200, 300],
}

WALK_FORWARD_WINDOWS = {
    "calibration": ("2008-01-01", "2015-12-31"),
    "validation": ("2016-01-01", "2019-12-31"),
    "test": ("2020-01-01", "2026-12-31"),
}

REGIME_WINDOWS = {
    "2008_financial_crisis": ("2008-01-01", "2008-12-31"),
    "2011_euro_debt": ("2011-01-01", "2011-12-31"),
    "2015_china_crash": ("2015-06-01", "2015-12-31"),
    "2018_trade_war": ("2018-01-01", "2018-12-31"),
    "2020_covid": ("2020-01-01", "2020-12-31"),
    "2022_rate_hike_bear": ("2022-01-01", "2022-12-31"),
    "2024_2026_ai_valuation": ("2024-01-01", "2026-12-31"),
}

