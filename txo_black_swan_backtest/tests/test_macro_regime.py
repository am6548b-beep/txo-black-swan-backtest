from __future__ import annotations

import pandas as pd

from src.indicators import add_market_indicators
from src.macro_regime import add_macro_regime_indicators, macro_restrictions
from src.stress_tests import synthetic_macro_factors, synthetic_market


def test_ai_supply_distortion_regime_detected() -> None:
    market = add_market_indicators(synthetic_market("G_ai_supply_distortion"))
    macro = synthetic_macro_factors(market, "G_ai_supply_distortion")
    out = add_macro_regime_indicators(market, macro)
    assert (out["macro_state"] == "AI_SUPPLY_DISTORTION").any()
    assert out["SupplyStressIndex"].max() > 60


def test_bullwhip_collapse_regime_detected() -> None:
    market = add_market_indicators(synthetic_market("H_bullwhip_collapse"))
    macro = synthetic_macro_factors(market, "H_bullwhip_collapse")
    out = add_macro_regime_indicators(market, macro)
    assert (out["macro_state"] == "BULLWHIP_COLLAPSE").any()


def test_extreme_combined_risk_flag() -> None:
    market = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=260),
            "tx_close": 100.0,
            "tx_open": 100.0,
            "tx_high": 101.0,
            "tx_low": 99.0,
            "txf_close": 100.0,
            "volume": 1,
            "vix": 20.0,
            "event_flag": 0,
        }
    )
    market = add_market_indicators(market)
    macro = pd.DataFrame(
        {
            "date": market["date"],
            "hbm_asp_index": pd.Series(range(260)).map(lambda x: 120 + x * 0.50),
            "ddr5_spot_index": pd.Series(range(260)).map(lambda x: 100 + x * 0.01),
            "pc_shipments_yoy": -12,
            "smartphone_shipments_yoy": -10,
            "pc_sellthrough_yoy": -15,
            "inventory_days_oem": 95,
            "inventory_days_components": 120,
            "pcb_revenue_yoy": 45,
            "mlcc_revenue_yoy": 45,
            "driver_ic_revenue_yoy": 45,
            "unit_growth_yoy": 2,
            "asp_growth_yoy": 35,
            "ai_server_capex_yoy": 80,
            "consumer_sentiment": pd.Series(range(260)).map(lambda x: 100 - x * 0.12),
            "cpi_yoy": 7,
            "core_cpi_yoy": 6,
            "ppi_yoy": 12,
            "real_wage_growth_yoy": -3,
            "consumer_confidence": pd.Series(range(260)).map(lambda x: 100 - x * 0.14),
            "unemployment_rate": pd.Series(range(260)).map(lambda x: 4.0 + x * 0.01),
            "policy_rate": 5,
            "us10y_yield": 5,
            "credit_card_delinquency": 4,
            "oil_price_yoy": 50,
            "usd_index": 110,
            "retail_sales_yoy": -3,
            "valuation_risk_index": 85,
            "liquidity_stress_index": 80,
        }
    )
    out = add_macro_regime_indicators(market, macro)
    assert out["AI_BULLWHIP_STAGFLATION_RISK"].iloc[-1] == "EXTREME"


def test_macro_restrictions_block_ic_in_collapse() -> None:
    assert macro_restrictions("BULLWHIP_COLLAPSE")["block_new_ic"] is True
    assert macro_restrictions("STAGFLATION_DEMAND_BREAK")["block_new_ic"] is True
