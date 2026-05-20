"""Synthetic stress scenarios for fragility testing."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from .backtester import run_backtest
from .metrics import max_drawdown


SCENARIOS = {
    "A_mild_correction": "1-month -10%, VIX 18 to 28, then +5%",
    "B_standard_black_swan": "10-day -22%, VIX 15 to 45, then +8%",
    "C_cascading_crash": "Down 15%, +3%, down 18%, no panic relief",
    "D_v_reversal": "Down 18%, then +15% in 7 days",
    "E_ai_bubble_ipo_liquidity": "Index -20%, portfolio beta temporarily 1.8-2.2",
    "F_carry_unwind_gap": "Gap risk, 3-5x bid/ask, 4x slippage proxy",
    "G_ai_supply_distortion": "HBM +80%, DDR5 +60%, components +40%, units +5%, sentiment weakens",
    "H_bullwhip_collapse": "PC -15%, smartphone -12%, inventory spike, components collapse, TX -18%, VIX jump",
}


def run_stress_suite(config: dict, put_params: dict, ic_params: dict) -> pd.DataFrame:
    rows = []
    for name in SCENARIOS:
        market = synthetic_market(name)
        macro = synthetic_macro_factors(market, name)
        options = synthetic_options(market, config, wide_spread=name.startswith("F_"))
        portfolio = synthetic_portfolio(market, config, high_beta=name.startswith("E_"))
        with TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            market.to_csv(data_dir / "market.csv", index=False)
            macro.to_csv(data_dir / "macro_factors.csv", index=False)
            options.to_csv(data_dir / "options.csv", index=False)
            portfolio.to_csv(data_dir / "portfolio.csv", index=False)
            cfg = config.copy()
            if name.startswith("F_"):
                cfg["normal_slippage_pct"] = config["normal_slippage_pct"] * 4
                cfg["stress_slippage_pct"] = config["stress_slippage_pct"] * 4
            equity, trades, _ = run_backtest(data_dir, cfg, put_params, ic_params, mode="full")
        if equity.empty:
            rows.append({"scenario": name, "note": "no_data"})
            continue
        warning_days = int((equity["warning"] == "margin_usage_above_limit").sum()) if "warning" in equity else 0
        forced_exits = trades[trades["reason"].str.contains("stop|touched|max_loss", regex=True, na=False)] if not trades.empty else trades
        ic_dates = trades[(trades["strategy"] == "iron_condor") & (trades["reason"] == "open_iron_condor")]["date"] if not trades.empty else pd.Series(dtype=str)
        rows.append(
            {
                "scenario": name,
                "description": SCENARIOS[name],
                "ending_equity": float(equity["total_equity"].iloc[-1]),
                "option_pnl": float(trades["cash_flow"].sum()) if not trades.empty else 0.0,
                "stock_pnl": float(equity["stock_equity"].iloc[-1] - equity["stock_equity"].iloc[0]),
                "max_drawdown": max_drawdown(equity["total_equity"]),
                "minimum_free_cash": float(equity["free_cash"].min()),
                "margin_warning_days": warning_days,
                "forced_exit_days": int(forced_exits["date"].nunique()) if not forced_exits.empty else 0,
                "ic_entered_too_early": bool(name.startswith(("C_", "D_")) and len(ic_dates) > 0),
                "entered_ai_supply_distortion": bool("macro_state" in equity and (equity["macro_state"] == "AI_SUPPLY_DISTORTION").any()),
                "entered_bullwhip_collapse": bool("macro_state" in equity and (equity["macro_state"] == "BULLWHIP_COLLAPSE").any()),
                "put_spread_opened": bool(not trades.empty and (trades["strategy"] == "put_spread").any()),
            }
        )
    rows.extend(run_delay_crash_tests(config, put_params, ic_params))
    return pd.DataFrame(rows)


def run_delay_crash_tests(config: dict, put_params: dict, ic_params: dict) -> list[dict]:
    """Check whether repeated hedge carry is survivable when the crash arrives late."""

    rows = []
    for months in [6, 9, 12]:
        market, crash_date = synthetic_delay_crash_market(months)
        macro = synthetic_macro_factors(market, "B_standard_black_swan")
        options = synthetic_options(market, config)
        portfolio = synthetic_portfolio(market, config)
        with TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            market.to_csv(data_dir / "market.csv", index=False)
            macro.to_csv(data_dir / "macro_factors.csv", index=False)
            options.to_csv(data_dir / "options.csv", index=False)
            portfolio.to_csv(data_dir / "portfolio.csv", index=False)
            equity, trades, _ = run_backtest(data_dir, config, put_params, ic_params, mode="full")
        pre_crash_trades = trades[pd.to_datetime(trades["date"]) < crash_date] if not trades.empty else trades
        hedge_buys = pre_crash_trades[(pre_crash_trades["strategy"] == "put_spread") & (pre_crash_trades["action"] == "BUY")] if not pre_crash_trades.empty else pre_crash_trades
        rows.append(
            {
                "scenario": f"Delay_crash_{months}m",
                "description": f"Risk is early, black swan arrives {months} months late",
                "ending_equity": float(equity["total_equity"].iloc[-1]) if not equity.empty else 0.0,
                "option_pnl": float(trades["cash_flow"].sum()) if not trades.empty else 0.0,
                "stock_pnl": float(equity["stock_equity"].iloc[-1] - equity["stock_equity"].iloc[0]) if not equity.empty else 0.0,
                "max_drawdown": max_drawdown(equity["total_equity"]) if not equity.empty else float("nan"),
                "minimum_free_cash": float(equity["free_cash"].min()) if not equity.empty else float("nan"),
                "margin_warning_days": int((equity["warning"] == "margin_usage_above_limit").sum()) if not equity.empty and "warning" in equity else 0,
                "forced_exit_days": 0,
                "ic_entered_too_early": False,
                "premium_drag_before_crash": -float(hedge_buys["cash_flow"].sum()) if not hedge_buys.empty else 0.0,
            }
        )
    return rows


def synthetic_market(name: str) -> pd.DataFrame:
    dates = pd.bdate_range("2021-01-04", periods=900)
    crash_start = 760
    price = np.full(len(dates), 18_000.0)
    vix = np.full(len(dates), 25.0)
    price[: crash_start - 126] = np.linspace(18_000, 19_000, crash_start - 126)
    price[crash_start - 126 : crash_start] = np.linspace(19_000, 24_000, 126)
    vix[crash_start - 220 : crash_start] = np.linspace(22, 15, 220)
    if name.startswith("A_"):
        price[crash_start : crash_start + 22] = np.linspace(price[crash_start - 1], price[crash_start - 1] * 0.90, 22)
        price[crash_start + 22 :] = np.linspace(price[crash_start + 21], price[crash_start + 21] * 1.05, len(dates) - crash_start - 22)
        vix[crash_start : crash_start + 22] = np.linspace(18, 28, 22)
        vix[crash_start + 22 :] = np.linspace(28, 20, len(dates) - crash_start - 22)
    elif name.startswith("B_"):
        price[crash_start : crash_start + 10] = np.linspace(price[crash_start - 1], price[crash_start - 1] * 0.78, 10)
        price[crash_start + 10 :] = np.linspace(price[crash_start + 9], price[crash_start + 9] * 1.08, len(dates) - crash_start - 10)
        vix[crash_start : crash_start + 10] = np.linspace(15, 45, 10)
        vix[crash_start + 10 :] = np.linspace(45, 25, len(dates) - crash_start - 10)
    elif name.startswith("C_"):
        price[crash_start : crash_start + 10] = np.linspace(price[crash_start - 1], price[crash_start - 1] * 0.85, 10)
        price[crash_start + 10 : crash_start + 15] = np.linspace(price[crash_start + 9], price[crash_start + 9] * 1.03, 5)
        price[crash_start + 15 : crash_start + 30] = np.linspace(price[crash_start + 14], price[crash_start - 1] * 0.70, 15)
        price[crash_start + 30 :] = price[crash_start + 29]
        vix[crash_start:] = 42
    elif name.startswith("D_"):
        price[crash_start : crash_start + 12] = np.linspace(price[crash_start - 1], price[crash_start - 1] * 0.82, 12)
        price[crash_start + 12 : crash_start + 19] = np.linspace(price[crash_start + 11], price[crash_start + 11] * 1.15, 7)
        price[crash_start + 19 :] = price[crash_start + 18]
        vix[crash_start : crash_start + 12] = np.linspace(16, 42, 12)
        vix[crash_start + 12 :] = np.linspace(42, 22, len(dates) - crash_start - 12)
    elif name.startswith("G_"):
        price[crash_start:] = np.linspace(price[crash_start - 1], price[crash_start - 1] * 1.05, len(dates) - crash_start)
        vix[crash_start:] = np.linspace(16, 20, len(dates) - crash_start)
    elif name.startswith("H_"):
        price[crash_start : crash_start + 20] = np.linspace(price[crash_start - 1], price[crash_start - 1] * 0.82, 20)
        price[crash_start + 20 :] = price[crash_start + 19]
        vix[crash_start : crash_start + 20] = np.linspace(16, 45, 20)
        vix[crash_start + 20 :] = np.linspace(45, 32, len(dates) - crash_start - 20)
    else:
        price[crash_start : crash_start + 15] = np.linspace(price[crash_start - 1], price[crash_start - 1] * 0.80, 15)
        price[crash_start + 15 :] = price[crash_start + 14]
        vix[crash_start:] = np.linspace(18, 48, len(dates) - crash_start)
    out = pd.DataFrame(
        {
            "date": dates,
            "tx_open": price,
            "tx_high": price * 1.01,
            "tx_low": price * 0.99,
            "tx_close": price,
            "txf_close": price,
            "volume": 100_000,
            "vix": vix,
            "event_flag": 0,
        }
    )
    return out


def synthetic_macro_factors(market: pd.DataFrame, name: str) -> pd.DataFrame:
    dates = pd.to_datetime(market["date"])
    n = len(dates)
    crash_start = min(760, n - 120)
    normal = pd.DataFrame(
        {
            "date": dates,
            "hbm_asp_index": np.linspace(100, 120, n),
            "ddr5_spot_index": np.linspace(100, 105, n),
            "pc_shipments_yoy": 2.0,
            "smartphone_shipments_yoy": 1.0,
            "pc_sellthrough_yoy": 2.0,
            "inventory_days_oem": 60.0,
            "inventory_days_components": 60.0,
            "pcb_revenue_yoy": 5.0,
            "mlcc_revenue_yoy": 5.0,
            "driver_ic_revenue_yoy": 5.0,
            "unit_growth_yoy": 3.0,
            "asp_growth_yoy": 4.0,
            "ai_server_capex_yoy": 20.0,
            "consumer_sentiment": 100.0,
            "cpi_yoy": 2.5,
            "core_cpi_yoy": 2.4,
            "ppi_yoy": 3.0,
            "real_wage_growth_yoy": 1.0,
            "consumer_confidence": 100.0,
            "unemployment_rate": 4.0,
            "policy_rate": 2.5,
            "us10y_yield": 3.0,
            "credit_card_delinquency": 2.0,
            "oil_price_yoy": 0.0,
            "usd_index": 100.0,
            "retail_sales_yoy": 3.5,
            "sox_relative_strength": 5.0,
            "tsmc_relative_strength": 5.0,
            "memory_relative_strength": 5.0,
            "pcb_relative_strength": 5.0,
            "mlcc_relative_strength": 5.0,
            "valuation_risk_index": 55.0,
            "liquidity_stress_index": 35.0,
            "event_flag": 0,
        }
    )
    if name.startswith("G_"):
        normal.loc[crash_start:, "hbm_asp_index"] = np.linspace(120, 180, n - crash_start)
        normal.loc[crash_start:, "ddr5_spot_index"] = np.linspace(105, 168, n - crash_start)
        component_path = np.linspace(20, 40, n - crash_start)
        for col in ["pcb_revenue_yoy", "mlcc_revenue_yoy", "driver_ic_revenue_yoy"]:
            normal.loc[crash_start:, col] = component_path
        normal.loc[crash_start:, "unit_growth_yoy"] = 5.0
        normal.loc[crash_start:, "asp_growth_yoy"] = np.linspace(15, 35, n - crash_start)
        normal.loc[crash_start:, "inventory_days_components"] = np.linspace(70, 95, n - crash_start)
        normal.loc[crash_start:, "pc_sellthrough_yoy"] = np.linspace(1, -2, n - crash_start)
        normal.loc[crash_start:, "consumer_sentiment"] = np.linspace(95, 78, n - crash_start)
        normal.loc[crash_start:, "valuation_risk_index"] = np.linspace(65, 80, n - crash_start)
    if name.startswith("H_"):
        normal.loc[:crash_start, "hbm_asp_index"] = np.linspace(100, 180, crash_start + 1)
        normal.loc[:crash_start, "ddr5_spot_index"] = np.linspace(100, 160, crash_start + 1)
        component_path = np.linspace(10, 40, crash_start + 1)
        for col in ["pcb_revenue_yoy", "mlcc_revenue_yoy", "driver_ic_revenue_yoy"]:
            normal.loc[:crash_start, col] = component_path
        normal.loc[crash_start:, "pc_shipments_yoy"] = -15.0
        normal.loc[crash_start:, "smartphone_shipments_yoy"] = -12.0
        normal.loc[crash_start:, "pc_sellthrough_yoy"] = -18.0
        normal.loc[crash_start:, "inventory_days_oem"] = np.linspace(90, 130, n - crash_start)
        normal.loc[crash_start:, "inventory_days_components"] = np.linspace(100, 150, n - crash_start)
        component_path = np.linspace(-5, -35, n - crash_start)
        for col in ["pcb_revenue_yoy", "mlcc_revenue_yoy", "driver_ic_revenue_yoy"]:
            normal.loc[crash_start:, col] = component_path
        normal.loc[crash_start:, "ddr5_spot_index"] = np.linspace(160, 110, n - crash_start)
        normal.loc[crash_start:, "consumer_sentiment"] = np.linspace(80, 55, n - crash_start)
        normal.loc[crash_start:, "valuation_risk_index"] = 85.0
        normal.loc[crash_start:, "liquidity_stress_index"] = np.linspace(75, 95, n - crash_start)
    return normal


def synthetic_delay_crash_market(months: int) -> tuple[pd.DataFrame, pd.Timestamp]:
    delay_days = months * 21
    crash_start = 760 + delay_days
    dates = pd.bdate_range("2021-01-04", periods=crash_start + 120)
    price = np.full(len(dates), 24_000.0)
    vix = np.full(len(dates), 25.0)
    price[: 760 - 126] = np.linspace(18_000, 19_000, 760 - 126)
    price[760 - 126 : 760] = np.linspace(19_000, 24_000, 126)
    price[760:crash_start] = np.linspace(24_000, 24_600, delay_days)
    vix[760 - 220 : crash_start] = np.linspace(22, 14, 220 + delay_days)
    price[crash_start : crash_start + 10] = np.linspace(price[crash_start - 1], price[crash_start - 1] * 0.78, 10)
    price[crash_start + 10 :] = np.linspace(price[crash_start + 9], price[crash_start + 9] * 1.08, len(dates) - crash_start - 10)
    vix[crash_start : crash_start + 10] = np.linspace(15, 45, 10)
    vix[crash_start + 10 :] = np.linspace(45, 25, len(dates) - crash_start - 10)
    out = pd.DataFrame(
        {
            "date": dates,
            "tx_open": price,
            "tx_high": price * 1.01,
            "tx_low": price * 0.99,
            "tx_close": price,
            "txf_close": price,
            "volume": 100_000,
            "vix": vix,
            "event_flag": 0,
        }
    )
    return out, pd.Timestamp(dates[crash_start])


def synthetic_options(market: pd.DataFrame, config: dict, wide_spread: bool = False) -> pd.DataFrame:
    rows = []
    spread = 0.20 if wide_spread else 0.08
    start = pd.Timestamp(market["date"].min()) + pd.offsets.BDay(30)
    end = pd.Timestamp(market["date"].max()) + pd.offsets.BDay(130)
    expiry_calendar = pd.bdate_range(start, end, freq="20B")
    global_low_strike = int(np.floor(float(market["txf_close"].min()) * 0.60 / 500) * 500)
    global_high_strike = int(np.ceil(float(market["txf_close"].max()) * 1.40 / 500) * 500)
    for row in market.itertuples(index=False):
        date = pd.Timestamp(row.date)
        if (date - pd.Timestamp(market["date"].min())).days < 600:
            continue
        expiries = [e for e in expiry_calendar if 20 <= np.busday_count(date.date(), pd.Timestamp(e).date()) <= 125][:3]
        for expiry in expiries:
            dte = int(np.busday_count(date.date(), pd.Timestamp(expiry).date()))
            time_scale = max(dte / 90.0, 0.05)
            for strike in range(global_low_strike, global_high_strike + 500, 500):
                for cp in ["P", "C"]:
                    iv = max(0.12, float(row.vix) / 100.0 * (1.1 if cp == "P" else 0.95))
                    intrinsic = max(strike - float(row.txf_close), 0.0) if cp == "P" else max(float(row.txf_close) - strike, 0.0)
                    distance = abs(strike / float(row.txf_close) - 1.0)
                    close = intrinsic + max(2.0, float(row.txf_close) * iv * np.exp(-distance * 8.0) * 0.018 * time_scale)
                    delta = (
                        -max(0.01, min(0.95, 1.0 / (1.0 + np.exp((float(row.txf_close) - strike) / 900.0))))
                        if cp == "P"
                        else max(0.01, min(0.95, 1.0 / (1.0 + np.exp((strike - float(row.txf_close)) / 900.0))))
                    )
                    rows.append(
                        {
                            "date": row.date,
                            "expiry": expiry,
                            "dte": dte,
                            "cp": cp,
                            "strike": strike,
                            "close": close,
                            "bid": close * (1 - spread),
                            "ask": close * (1 + spread),
                            "volume": 200,
                            "open_interest": 300,
                            "iv": iv,
                            "delta": delta,
                        }
                    )
    return pd.DataFrame(rows)


def synthetic_portfolio(market: pd.DataFrame, config: dict, high_beta: bool = False) -> pd.DataFrame:
    beta = float(config["portfolio_beta"])
    returns = market["tx_close"].pct_change().fillna(0.0)
    equity = [float(config["initial_stock_equity"])]
    betas = []
    for i, r in enumerate(returns):
        b = 2.0 if high_beta and i >= 740 else beta
        betas.append(b)
        if i > 0:
            equity.append(equity[-1] * (1.0 + b * float(r)))
    return pd.DataFrame({"date": market["date"], "stock_equity": equity, "portfolio_beta": betas})
