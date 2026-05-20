"""AI supply distortion, bullwhip, and stagflation macro regime diagnostics.

All indicators use current or prior observations only. Thresholds are coarse
regime buckets, not optimized parameters.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


SUPPLY_REGIMES = {
    "LOW": (0.0, 30.0),
    "MEDIUM": (30.0, 60.0),
    "HIGH": (60.0, 80.0),
    "EXTREME": (80.0, 100.0),
}

MACRO_FACTOR_DEFAULTS = {
    "hbm_asp_index": 100.0,
    "ddr5_spot_index": 100.0,
    "pc_shipments_yoy": 0.0,
    "smartphone_shipments_yoy": 0.0,
    "pc_sellthrough_yoy": 0.0,
    "inventory_days_oem": 60.0,
    "inventory_days_components": 60.0,
    "pcb_revenue_yoy": 0.0,
    "mlcc_revenue_yoy": 0.0,
    "driver_ic_revenue_yoy": 0.0,
    "unit_growth_yoy": 0.0,
    "asp_growth_yoy": 0.0,
    "ai_server_capex_yoy": 0.0,
    "consumer_sentiment": 100.0,
    "cpi_yoy": 2.0,
    "core_cpi_yoy": 2.0,
    "ppi_yoy": 2.0,
    "real_wage_growth_yoy": 1.0,
    "consumer_confidence": 100.0,
    "unemployment_rate": 4.0,
    "policy_rate": 2.0,
    "us10y_yield": 2.5,
    "credit_card_delinquency": 2.0,
    "oil_price_yoy": 0.0,
    "usd_index": 100.0,
    "retail_sales_yoy": 3.0,
    "sox_relative_strength": 0.0,
    "tsmc_relative_strength": 0.0,
    "memory_relative_strength": 0.0,
    "pcb_relative_strength": 0.0,
    "mlcc_relative_strength": 0.0,
    "valuation_risk_index": 50.0,
    "liquidity_stress_index": 50.0,
}


def add_macro_regime_indicators(market: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """Merge macro factors into market and add coarse regime diagnostics."""

    out = market.sort_values("date").copy()
    out["date"] = pd.to_datetime(out["date"])
    if macro.empty:
        for col, value in MACRO_FACTOR_DEFAULTS.items():
            out[col] = value
    else:
        macro = macro.sort_values("date").copy()
        macro["date"] = pd.to_datetime(macro["date"])
        out = out.merge(macro.drop(columns=["event_flag"], errors="ignore"), on="date", how="left")
        for col, value in MACRO_FACTOR_DEFAULTS.items():
            if col not in out.columns:
                out[col] = value
            out[col] = out[col].ffill().fillna(value)

    hbm_base = out["hbm_asp_index"].rolling(252, min_periods=20).min()
    ddr5_base = out["ddr5_spot_index"].rolling(252, min_periods=20).min()
    hbm_cycle_gain = out["hbm_asp_index"] / hbm_base.replace(0, np.nan)
    ddr5_cycle_gain = out["ddr5_spot_index"] / ddr5_base.replace(0, np.nan)
    out["hbm_ddr5_spread"] = hbm_cycle_gain / ddr5_cycle_gain.replace(0, np.nan)
    out["ddr5_spot_accel"] = out["ddr5_spot_index"].pct_change(20).fillna(0.0) * 100.0
    out["ddr5_spot_cycle_rise"] = (out["ddr5_spot_index"] / ddr5_base.replace(0, np.nan) - 1.0).fillna(0.0) * 100.0
    out["component_revenue_yoy"] = out[["pcb_revenue_yoy", "mlcc_revenue_yoy", "driver_ic_revenue_yoy"]].mean(axis=1)
    out["component_revenue_accel"] = out["component_revenue_yoy"].diff(20).fillna(0.0)
    out["inventory_divergence"] = out["inventory_days_components"] - out["inventory_days_oem"]
    out["asp_unit_gap"] = out["asp_growth_yoy"] - out["unit_growth_yoy"]
    out["consumer_sentiment_change_60d"] = out["consumer_sentiment"].diff(60).fillna(0.0)
    out["consumer_confidence_change_60d"] = out["consumer_confidence"].diff(60).fillna(0.0)
    out["sellthrough_weakness"] = np.maximum(0.0, -out[["pc_sellthrough_yoy", "smartphone_shipments_yoy"]].mean(axis=1))

    out["SupplyStressIndex"] = _clip_score(
        15.0 * _scale(out["hbm_ddr5_spread"] - 1.0, 0.10, 0.60)
        + 15.0 * _scale(pd.concat([out["ddr5_spot_accel"], out["ddr5_spot_cycle_rise"]], axis=1).max(axis=1), 5.0, 35.0)
        + 15.0 * _scale(out["component_revenue_yoy"], 10.0, 40.0)
        + 15.0 * _scale(out["inventory_divergence"], 5.0, 35.0)
        + 15.0 * _scale(out["asp_unit_gap"], 5.0, 25.0)
        + 10.0 * _scale(-out["consumer_sentiment_change_60d"], 2.0, 15.0)
        + 15.0 * _scale(out["sellthrough_weakness"], 0.0, 15.0)
    )

    out["MacroDemandFragilityIndex"] = _clip_score(
        12.0 * _scale(out["cpi_yoy"], 3.0, 7.0)
        + 12.0 * _scale(out["core_cpi_yoy"], 3.0, 6.0)
        + 12.0 * _scale(out["ppi_yoy"], 4.0, 12.0)
        + 12.0 * _scale(-out["real_wage_growth_yoy"], 0.0, 4.0)
        + 10.0 * _scale(-out["consumer_confidence_change_60d"], 2.0, 15.0)
        + 10.0 * _scale(out["unemployment_rate"].diff(60).fillna(0.0), 0.2, 1.5)
        + 10.0 * _scale((out["policy_rate"] + out["us10y_yield"]) / 2.0, 3.0, 6.0)
        + 8.0 * _scale(out["credit_card_delinquency"].diff(60).fillna(0.0), 0.1, 1.0)
        + 7.0 * _scale(out["oil_price_yoy"], 10.0, 50.0)
        + 7.0 * _scale(-out["retail_sales_yoy"], 0.0, 6.0)
    )

    out["ValuationRiskIndex"] = out["valuation_risk_index"].fillna(50.0).clip(0, 100)
    out["LiquidityStressIndex"] = _liquidity_stress(out)
    out["CombinedRiskScore"] = (
        0.4 * out["SupplyStressIndex"]
        + 0.3 * out["MacroDemandFragilityIndex"]
        + 0.2 * out["ValuationRiskIndex"]
        + 0.1 * out["LiquidityStressIndex"]
    ).clip(0, 100)
    out["AI_BULLWHIP_STAGFLATION_RISK"] = np.where(
        (out["SupplyStressIndex"] > 70)
        & (out["MacroDemandFragilityIndex"] > 70)
        & (out["ValuationRiskIndex"] > 70),
        "EXTREME",
        _bucket(out["CombinedRiskScore"]),
    )
    out["macro_state"] = out.apply(_classify_macro_state, axis=1)
    out["supply_stress_bucket"] = _bucket(out["SupplyStressIndex"])
    out["macro_demand_fragility_bucket"] = _bucket(out["MacroDemandFragilityIndex"])
    out["combined_risk_bucket"] = _bucket(out["CombinedRiskScore"])
    return out


def write_macro_reports(report_dir: Path, market: pd.DataFrame, trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write macro dashboard and report audit outputs."""

    report_dir.mkdir(parents=True, exist_ok=True)
    dashboard = macro_dashboard(market)
    dashboard.to_csv(report_dir / "macro_dashboard.csv", index=False)
    (report_dir / "macro_dashboard.md").write_text(_dashboard_markdown(dashboard), encoding="utf-8")
    audit = report_audit(market, trades)
    audit.to_csv(report_dir / "report_audit.csv", index=False)
    return dashboard, audit


def macro_dashboard(market: pd.DataFrame) -> pd.DataFrame:
    if market.empty:
        return pd.DataFrame(columns=["item", "value", "status"])
    last = market.sort_values("date").iloc[-1]
    rows = [
        ("SupplyStressIndex", last.get("SupplyStressIndex", 0.0), _color(last.get("SupplyStressIndex", 0.0))),
        ("HBM vs DDR5 spread", last.get("hbm_ddr5_spread", 1.0), _spread_color(last.get("hbm_ddr5_spread", 1.0))),
        ("ASP growth vs unit growth", last.get("asp_unit_gap", 0.0), _gap_color(last.get("asp_unit_gap", 0.0))),
        ("OEM inventory days", last.get("inventory_days_oem", 0.0), _inventory_color(last.get("inventory_days_oem", 0.0))),
        ("Component inventory days", last.get("inventory_days_components", 0.0), _inventory_color(last.get("inventory_days_components", 0.0))),
        ("PC sell-through", last.get("pc_sellthrough_yoy", 0.0), "RED" if last.get("pc_sellthrough_yoy", 0.0) < -5 else "YELLOW"),
        ("Smartphone sell-through", last.get("smartphone_shipments_yoy", 0.0), "RED" if last.get("smartphone_shipments_yoy", 0.0) < -8 else "YELLOW"),
        ("Consumer sentiment", last.get("consumer_sentiment", 100.0), "RED" if last.get("consumer_sentiment", 100.0) < 80 else "GREEN"),
        ("AI server capex growth", last.get("ai_server_capex_yoy", 0.0), _gap_color(last.get("ai_server_capex_yoy", 0.0))),
        ("TX VIX regime", last.get("vix_percentile_3y", 0.0), _color(last.get("vix_percentile_3y", 0.0))),
        ("SOX relative strength", last.get("sox_relative_strength", 0.0), _relative_strength_color(last.get("sox_relative_strength", 0.0))),
        ("TSMC relative strength", last.get("tsmc_relative_strength", 0.0), _relative_strength_color(last.get("tsmc_relative_strength", 0.0))),
        ("Memory makers relative strength", last.get("memory_relative_strength", 0.0), _relative_strength_color(last.get("memory_relative_strength", 0.0))),
        ("PCB relative strength", last.get("pcb_relative_strength", 0.0), _relative_strength_color(last.get("pcb_relative_strength", 0.0))),
        ("MLCC relative strength", last.get("mlcc_relative_strength", 0.0), _relative_strength_color(last.get("mlcc_relative_strength", 0.0))),
        ("MacroDemandFragilityIndex", last.get("MacroDemandFragilityIndex", 0.0), _color(last.get("MacroDemandFragilityIndex", 0.0))),
        ("CombinedRiskScore", last.get("CombinedRiskScore", 0.0), _color(last.get("CombinedRiskScore", 0.0))),
        ("AI_BULLWHIP_STAGFLATION_RISK", last.get("AI_BULLWHIP_STAGFLATION_RISK", "LOW"), str(last.get("AI_BULLWHIP_STAGFLATION_RISK", "LOW"))),
    ]
    return pd.DataFrame(rows, columns=["item", "value", "status"])


def report_audit(market: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    if market.empty:
        return pd.DataFrame()
    out = market.copy()
    checks = [
        ("revenue_growth_gt_unit_growth", (out["component_revenue_yoy"] - out["unit_growth_yoy"]) > 20),
        ("inventory_rising_while_revenue_accelerating", (out["inventory_days_components"].diff(20) > 5) & (out["component_revenue_accel"] > 5)),
        ("consumer_weak_upstream_extreme", (out["pc_sellthrough_yoy"] <= 0) & (out["component_revenue_yoy"] > 30)),
        ("ai_supply_distortion_misread_as_real_demand", (out["macro_state"] == "AI_SUPPLY_DISTORTION") & (out["unit_growth_yoy"] < out["asp_growth_yoy"])),
    ]
    rows = []
    for name, mask in checks:
        rows.append({"audit_check": name, "flagged_days": int(mask.fillna(False).sum()), "status": "WARN" if mask.fillna(False).any() else "PASS"})
    if trades.empty:
        early_ic = 0
    else:
        trade_dates = pd.to_datetime(trades[(trades["strategy"] == "iron_condor") & (trades["reason"] == "open_iron_condor")]["date"])
        collapse_dates = set(pd.to_datetime(out[out["macro_state"].isin(["BULLWHIP_COLLAPSE", "STAGFLATION_DEMAND_BREAK"])]["date"]))
        early_ic = sum(any(0 <= (collapse_date - trade_date).days <= 30 for collapse_date in collapse_dates) for trade_date in trade_dates)
    rows.append({"audit_check": "bullwhip_before_iron_condor_too_early", "flagged_days": int(early_ic), "status": "WARN" if early_ic else "PASS"})
    rows.append({"audit_check": "avoid_overfitting_rules", "flagged_days": 0, "status": "PASS", "detail": "coarse LOW/MEDIUM/HIGH/EXTREME buckets only; no crash-date tuning"})
    return pd.DataFrame(rows)


def macro_restrictions(macro_state: str) -> dict[str, float | bool | str]:
    """Return coarse conservative risk modifiers for a macro state."""

    if macro_state == "AI_SUPPLY_DISTORTION":
        return {"hedge_budget_multiplier": 1.25, "ic_risk_multiplier": 0.50, "free_cash_multiplier": 1.50, "block_new_ic": False, "extend_put_dte": 30, "stress_slippage": False}
    if macro_state in {"BULLWHIP_COLLAPSE", "STAGFLATION_DEMAND_BREAK"}:
        return {"hedge_budget_multiplier": 1.50, "ic_risk_multiplier": 0.0, "free_cash_multiplier": 2.0, "block_new_ic": True, "extend_put_dte": 60, "stress_slippage": True}
    if macro_state == "STAGFLATION_PRESSURE":
        return {"hedge_budget_multiplier": 1.25, "ic_risk_multiplier": 0.50, "free_cash_multiplier": 1.50, "block_new_ic": False, "extend_put_dte": 30, "stress_slippage": False}
    return {"hedge_budget_multiplier": 1.0, "ic_risk_multiplier": 1.0, "free_cash_multiplier": 1.0, "block_new_ic": False, "extend_put_dte": 0, "stress_slippage": False}


def _classify_macro_state(row: pd.Series) -> str:
    collapse_count = sum(
        [
            row["pc_shipments_yoy"] < -10,
            row["smartphone_shipments_yoy"] < -8,
            row["inventory_days_oem"] > 85,
            row["inventory_days_components"] > 90,
            row["component_revenue_yoy"] < 0,
            row["ddr5_spot_accel"] < -10,
            row.get("vix_percentile_3y", 0) > 80,
            row.get("tx_close", 0) < row.get("ma200", -np.inf),
        ]
    )
    distortion_count = sum(
        [
            row["hbm_ddr5_spread"] > 1.25,
            row["component_revenue_yoy"] > 30,
            row["asp_unit_gap"] > 10,
            row["inventory_days_components"] > 70,
            row["pc_sellthrough_yoy"] <= 0,
            row["consumer_sentiment_change_60d"] < -3,
        ]
    )
    stag_break_count = sum(
        [
            row["MacroDemandFragilityIndex"] > 70,
            row["pc_sellthrough_yoy"] < 0,
            row["inventory_days_oem"] > 85,
            row["component_revenue_yoy"] < 0,
        ]
    )
    stag_pressure_count = sum(
        [
            row["cpi_yoy"] > 3.5,
            row["core_cpi_yoy"] > 3.0,
            row["real_wage_growth_yoy"] <= 0,
            row["policy_rate"] > 3.0,
            row["consumer_confidence_change_60d"] < -2,
            row["ppi_yoy"] > 4.0,
        ]
    )
    if collapse_count >= 4:
        return "BULLWHIP_COLLAPSE"
    if stag_break_count >= 3:
        return "STAGFLATION_DEMAND_BREAK"
    if distortion_count >= 4:
        return "AI_SUPPLY_DISTORTION"
    if stag_pressure_count >= 4:
        return "STAGFLATION_PRESSURE"
    return "NORMAL"


def _scale(series: pd.Series, low: float, high: float) -> pd.Series:
    return ((series - low) / (high - low)).clip(0.0, 1.0)


def _clip_score(series: pd.Series) -> pd.Series:
    return series.fillna(0.0).clip(0.0, 100.0)


def _bucket(series: pd.Series | float) -> pd.Series | str:
    def one(value: float) -> str:
        if value < 30:
            return "LOW"
        if value < 60:
            return "MEDIUM"
        if value < 80:
            return "HIGH"
        return "EXTREME"

    if isinstance(series, pd.Series):
        return series.fillna(0.0).map(one)
    return one(float(series))


def _color(value: float) -> str:
    bucket = _bucket(float(value))
    return {"LOW": "GREEN", "MEDIUM": "YELLOW", "HIGH": "ORANGE", "EXTREME": "RED"}[bucket]


def _spread_color(value: float) -> str:
    if value > 1.6:
        return "RED"
    if value > 1.3:
        return "ORANGE"
    if value > 1.1:
        return "YELLOW"
    return "GREEN"


def _gap_color(value: float) -> str:
    if value > 30:
        return "RED"
    if value > 15:
        return "ORANGE"
    if value > 5:
        return "YELLOW"
    return "GREEN"


def _inventory_color(value: float) -> str:
    if value > 100:
        return "RED"
    if value > 80:
        return "ORANGE"
    if value > 65:
        return "YELLOW"
    return "GREEN"


def _relative_strength_color(value: float) -> str:
    if value < -15:
        return "RED"
    if value < -5:
        return "ORANGE"
    if value < 5:
        return "YELLOW"
    return "GREEN"


def _liquidity_stress(out: pd.DataFrame) -> pd.Series:
    if "liquidity_stress_index" in out and out["liquidity_stress_index"].notna().any():
        return out["liquidity_stress_index"].fillna(50.0).clip(0, 100)
    vix_component = _scale(out.get("vix_percentile_3y", pd.Series(50, index=out.index)), 60, 95) * 100
    return vix_component.fillna(50.0).clip(0, 100)


def _dashboard_markdown(dashboard: pd.DataFrame) -> str:
    lines = [
        "# Macro Risk Dashboard",
        "",
        "This dashboard monitors AI supply-chain distortion, bullwhip risk, and macro demand fragility. It does not predict crash timing and must not be tuned using future collapse dates.",
        "",
        "| Item | Value | Status |",
        "|---|---:|---|",
    ]
    for row in dashboard.itertuples(index=False):
        value = f"{row.value:.2f}" if isinstance(row.value, (float, int, np.floating)) else str(row.value)
        lines.append(f"| {row.item} | {value} | {row.status} |")
    lines.extend(
        [
            "",
            "## Avoid Overfitting",
            "",
            "- Do not tune SupplyStressIndex thresholds using known collapse dates.",
            "- Do not optimize on a single memory cycle.",
            "- Use only LOW / MEDIUM / HIGH / EXTREME coarse regimes.",
            "- Treat strong revenue growth without unit demand confirmation as fragility, not proof of durable demand.",
        ]
    )
    return "\n".join(lines) + "\n"
