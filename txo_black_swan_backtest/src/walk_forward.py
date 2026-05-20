"""Walk-forward and coarse robustness checks."""

from __future__ import annotations

import itertools
from pathlib import Path

import pandas as pd

from .backtester import run_backtest
from .metrics import summarize


def coarse_grid(grid: dict) -> list[dict]:
    keys = list(grid.keys())
    rows = []
    for values in itertools.product(*(grid[k] for k in keys)):
        rows.append(dict(zip(keys, values)))
    return rows


def apply_grid_params(base_put: dict, base_ic: dict, row: dict) -> tuple[dict, dict]:
    put = base_put.copy()
    ic = base_ic.copy()
    put["long_put_moneyness"] = row["long_put_moneyness"]
    put["short_put_moneyness"] = row["short_put_moneyness"]
    put["target_dte_min"], put["target_dte_max"] = row["put_spread_dte"]
    ic["short_put_delta"] = -abs(row["iron_condor_put_delta"])
    ic["short_call_delta"] = row["iron_condor_call_delta"]
    ic["wing_width"] = row["wing_width"]
    return put, ic


def run_walk_forward(
    data_dir: Path,
    config: dict,
    base_put: dict,
    base_ic: dict,
    grid: dict,
    windows: dict[str, tuple[str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    param_rows = coarse_grid(grid)
    for i, params in enumerate(param_rows, 1):
        put, ic = apply_grid_params(base_put, base_ic, params)
        equity, trades, market = run_backtest(data_dir, config, put, ic, mode="full")
        if equity.empty:
            rows.append({"param_id": i, **params, "note": "no_data"})
            continue
        equity["date"] = pd.to_datetime(equity["date"])
        for window, (start, end) in windows.items():
            sub = equity[(equity["date"] >= pd.Timestamp(start)) & (equity["date"] <= pd.Timestamp(end))]
            market_sub = market[(market["date"] >= pd.Timestamp(start)) & (market["date"] <= pd.Timestamp(end))] if not market.empty else market
            diagnostics = {
                "max_supply_stress_index": float(market_sub["SupplyStressIndex"].max()) if not market_sub.empty and "SupplyStressIndex" in market_sub else 0.0,
                "max_macro_demand_fragility_index": float(market_sub["MacroDemandFragilityIndex"].max()) if not market_sub.empty and "MacroDemandFragilityIndex" in market_sub else 0.0,
                "max_combined_risk_score": float(market_sub["CombinedRiskScore"].max()) if not market_sub.empty and "CombinedRiskScore" in market_sub else 0.0,
                "ai_supply_distortion_days": int((market_sub["macro_state"] == "AI_SUPPLY_DISTORTION").sum()) if not market_sub.empty and "macro_state" in market_sub else 0,
                "bullwhip_collapse_days": int((market_sub["macro_state"] == "BULLWHIP_COLLAPSE").sum()) if not market_sub.empty and "macro_state" in market_sub else 0,
                "stagflation_pressure_days": int((market_sub["macro_state"] == "STAGFLATION_PRESSURE").sum()) if not market_sub.empty and "macro_state" in market_sub else 0,
                "stagflation_demand_break_days": int((market_sub["macro_state"] == "STAGFLATION_DEMAND_BREAK").sum()) if not market_sub.empty and "macro_state" in market_sub else 0,
            }
            rows.append({"param_id": i, "window": window, **params, **summarize(sub, trades, f"param_{i}_{window}"), **diagnostics})
    summary = pd.DataFrame(rows)
    return summary, stability_report(summary)


def stability_report(summary: pd.DataFrame) -> pd.DataFrame:
    """Flag single-point results that look much better than the coarse neighborhood."""

    if summary.empty or "max_drawdown" not in summary:
        return pd.DataFrame()
    rows = []
    for pid, grp in summary.groupby("param_id"):
        mdd = grp["max_drawdown"].dropna()
        if mdd.empty:
            continue
        own = float(mdd.mean())
        peers = summary[summary["param_id"] != pid]["max_drawdown"].dropna()
        peer_median = float(peers.median()) if not peers.empty else own
        risk = "LOW"
        if own > peer_median * 0.5 and own > peer_median + 0.05:
            risk = "HIGH"
        elif own > peer_median + 0.025:
            risk = "MEDIUM"
        rows.append(
            {
                "param_id": pid,
                "mean_max_drawdown": own,
                "peer_median_max_drawdown": peer_median,
                "overfit_risk": risk,
                "note": "Do not choose by total return; inspect drawdown stability and cost sensitivity.",
            }
        )
    return pd.DataFrame(rows)
