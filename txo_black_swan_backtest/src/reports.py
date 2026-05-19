"""Report and chart writers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .metrics import regime_report, summarize, trade_pnl

TRADE_COLUMNS = ["date", "position_id", "strategy", "action", "cp", "strike", "expiry", "quantity", "price", "cash_flow", "cost", "reason"]
EQUITY_COLUMNS = ["date", "cash", "stock_equity", "option_value", "total_equity", "free_cash", "required_margin", "margin_usage", "state", "daily_stock_pnl", "daily_option_pnl", "warning"]


def write_reports(
    report_dir: Path,
    equity: pd.DataFrame,
    trades: pd.DataFrame,
    regimes: dict[str, tuple[str, str]],
    label: str = "base",
    overfit: pd.DataFrame | None = None,
    stress: pd.DataFrame | None = None,
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    charts = report_dir / "charts"
    charts.mkdir(exist_ok=True)
    pd.DataFrame([summarize(equity, trades, label)]).to_csv(report_dir / "summary.csv", index=False)
    (trades if not trades.empty else pd.DataFrame(columns=TRADE_COLUMNS)).to_csv(report_dir / "trades.csv", index=False)
    (equity if not equity.empty else pd.DataFrame(columns=EQUITY_COLUMNS)).to_csv(report_dir / "equity_curve.csv", index=False)
    regime_report(equity, regimes).to_csv(report_dir / "regime_report.csv", index=False)
    (stress if stress is not None else pd.DataFrame()).to_csv(report_dir / "stress_report.csv", index=False)
    (overfit if overfit is not None else pd.DataFrame()).to_csv(report_dir / "overfit_risk_report.csv", index=False)
    if not equity.empty:
        _plot_equity(equity, charts)
        _plot_drawdown(equity, charts)
        _plot_margin(equity, charts)
    if not trades.empty:
        _plot_hedge_cost(trades, charts)
        _plot_ic_distribution(trades, charts)
        _plot_put_payoff(trades, charts)


def _plot_equity(equity: pd.DataFrame, charts: Path) -> None:
    frame = equity.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    plt.figure(figsize=(10, 5))
    plt.plot(frame["date"], frame["total_equity"])
    plt.title("Total Equity Curve")
    plt.tight_layout()
    plt.savefig(charts / "total_equity_curve.png")
    plt.close()


def _plot_drawdown(equity: pd.DataFrame, charts: Path) -> None:
    frame = equity.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    dd = frame["total_equity"] / frame["total_equity"].cummax() - 1.0
    plt.figure(figsize=(10, 4))
    plt.plot(frame["date"], dd)
    plt.title("Drawdown Curve")
    plt.tight_layout()
    plt.savefig(charts / "drawdown_curve.png")
    plt.close()


def _plot_margin(equity: pd.DataFrame, charts: Path) -> None:
    frame = equity.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    plt.figure(figsize=(10, 4))
    plt.plot(frame["date"], frame["margin_usage"])
    plt.title("Margin Usage")
    plt.tight_layout()
    plt.savefig(charts / "margin_usage_curve.png")
    plt.close()


def _plot_hedge_cost(trades: pd.DataFrame, charts: Path) -> None:
    hedge = trades[(trades["strategy"] == "put_spread") & (trades["action"] == "BUY")].copy()
    if hedge.empty:
        return
    hedge["date"] = pd.to_datetime(hedge["date"])
    yearly = -hedge.groupby(hedge["date"].dt.year)["cash_flow"].sum()
    plt.figure(figsize=(8, 4))
    yearly.plot(kind="bar")
    plt.title("Hedge Cost By Year")
    plt.tight_layout()
    plt.savefig(charts / "hedge_cost_by_year.png")
    plt.close()


def _plot_ic_distribution(trades: pd.DataFrame, charts: Path) -> None:
    pnl = trade_pnl(trades)
    ic = pnl[pnl["strategy"] == "iron_condor"]
    if ic.empty:
        return
    plt.figure(figsize=(8, 4))
    plt.hist(ic["pnl"], bins=20)
    plt.title("Iron Condor PnL Distribution")
    plt.tight_layout()
    plt.savefig(charts / "iron_condor_pnl_distribution.png")
    plt.close()


def _plot_put_payoff(trades: pd.DataFrame, charts: Path) -> None:
    pnl = trade_pnl(trades)
    ps = pnl[pnl["strategy"] == "put_spread"]
    if ps.empty:
        return
    plt.figure(figsize=(8, 4))
    ps["pnl"].plot(kind="bar")
    plt.title("Put Spread Payoff During Crash")
    plt.tight_layout()
    plt.savefig(charts / "put_spread_payoff_during_crash.png")
    plt.close()
