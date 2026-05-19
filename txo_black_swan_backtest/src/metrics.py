"""Performance and robustness metrics."""

from __future__ import annotations

import pandas as pd


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return float("nan")
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def cagr(equity: pd.Series, dates: pd.Series) -> float:
    if equity.empty or len(equity) < 2:
        return float("nan")
    start = float(equity.iloc[0])
    end = float(equity.iloc[-1])
    days = max((pd.Timestamp(dates.iloc[-1]) - pd.Timestamp(dates.iloc[0])).days, 1)
    if start <= 0 or end <= 0:
        return float("nan")
    return float((end / start) ** (365.25 / days) - 1.0)


def worst_month(equity: pd.DataFrame) -> float:
    if equity.empty:
        return float("nan")
    frame = equity.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    try:
        monthly = frame.set_index("date")["total_equity"].resample("ME").last().pct_change().dropna()
    except ValueError:
        monthly = frame.set_index("date")["total_equity"].resample("M").last().pct_change().dropna()
    return float(monthly.min()) if not monthly.empty else float("nan")


def trade_pnl(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["position_id", "strategy", "pnl"])
    grouped = trades.groupby(["position_id", "strategy"], as_index=False)["cash_flow"].sum()
    grouped = grouped.rename(columns={"cash_flow": "pnl"})
    return grouped


def avg_win_loss(trades: pd.DataFrame) -> tuple[float, float]:
    pnl = trade_pnl(trades)
    wins = pnl[pnl["pnl"] > 0]["pnl"]
    losses = pnl[pnl["pnl"] < 0]["pnl"]
    return (
        float(wins.mean()) if not wins.empty else 0.0,
        float(losses.mean()) if not losses.empty else 0.0,
    )


def max_consecutive_losses(trades: pd.DataFrame) -> int:
    pnl = trade_pnl(trades)
    if pnl.empty:
        return 0
    streak = 0
    worst = 0
    for value in pnl["pnl"]:
        if value < 0:
            streak += 1
            worst = max(worst, streak)
        else:
            streak = 0
    return worst


def summarize(equity: pd.DataFrame, trades: pd.DataFrame, label: str = "base") -> dict:
    if equity.empty:
        return {"label": label, "note": "no_data"}
    option_pnl = float(trades["cash_flow"].sum()) if not trades.empty else 0.0
    pnl = trade_pnl(trades)
    ic = pnl[pnl["strategy"] == "iron_condor"] if not pnl.empty else pnl
    hedge_trades = trades[trades["strategy"] == "put_spread"] if not trades.empty else trades
    hedge_cost = -float(hedge_trades[hedge_trades["action"] == "BUY"]["cash_flow"].sum()) if not hedge_trades.empty else 0.0
    avg_win, avg_loss = avg_win_loss(trades)
    returns = equity["total_equity"].pct_change().fillna(0.0)
    turnover = float(trades["cash_flow"].abs().sum()) if not trades.empty else 0.0
    return {
        "label": label,
        "CAGR": cagr(equity["total_equity"], equity["date"]),
        "max_drawdown": max_drawdown(equity["total_equity"]),
        "worst_month": worst_month(equity),
        "worst_trade": float(pnl["pnl"].min()) if not pnl.empty else 0.0,
        "number_of_trades": int(pnl.shape[0]) if not pnl.empty else 0,
        "average_hedge_cost": hedge_cost / max(1, len(pd.to_datetime(equity["date"]).dt.year.unique())),
        "hedge_payoff_during_crash": option_pnl,
        "iron_condor_win_rate": float((ic["pnl"] > 0).mean()) if not ic.empty else 0.0,
        "average_win": avg_win,
        "average_loss": avg_loss,
        "max_consecutive_losses": max_consecutive_losses(trades),
        "minimum_free_cash": float(equity["free_cash"].min()),
        "margin_usage_max": float(equity["margin_usage"].max()),
        "turnover": turnover,
        "cost_drag": float(trades["cost"].sum()) if not trades.empty and "cost" in trades else 0.0,
        "daily_vol": float(returns.std()),
    }


def regime_report(equity: pd.DataFrame, regimes: dict[str, tuple[str, str]]) -> pd.DataFrame:
    rows = []
    if equity.empty:
        return pd.DataFrame(rows)
    eq = equity.copy()
    eq["date"] = pd.to_datetime(eq["date"])
    for name, (start, end) in regimes.items():
        sub = eq[(eq["date"] >= pd.Timestamp(start)) & (eq["date"] <= pd.Timestamp(end))]
        if sub.empty:
            rows.append({"regime": name, "note": "no_data"})
        else:
            rows.append(
                {
                    "regime": name,
                    "start": start,
                    "end": end,
                    "CAGR": cagr(sub["total_equity"], sub["date"]),
                    "max_drawdown": max_drawdown(sub["total_equity"]),
                    "worst_month": worst_month(sub),
                }
            )
    return pd.DataFrame(rows)
