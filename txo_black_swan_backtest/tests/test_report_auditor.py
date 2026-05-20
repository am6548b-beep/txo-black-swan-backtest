from __future__ import annotations

import pandas as pd

from src.report_auditor import run_report_audit


def test_report_audit_writes_csv_and_md(tmp_path) -> None:
    report_dir = tmp_path
    pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-02", "2024-01-10", "2024-01-10"],
            "position_id": ["PS-1"] * 4,
            "strategy": ["put_spread"] * 4,
            "action": ["BUY", "SELL", "SELL", "BUY"],
            "cp": ["P"] * 4,
            "strike": [9000, 7500, 9000, 7500],
            "expiry": ["2024-03-01"] * 4,
            "quantity": [1, -1, -1, 1],
            "price": [100, 30, 200, 20],
            "cash_flow": [-5030, 1470, 9970, -1030],
            "cost": [30, 30, 30, 30],
            "reason": ["open_put_spread", "open_put_spread", "put_spread_full_take_profit", "put_spread_full_take_profit"],
        }
    ).to_csv(report_dir / "trades.csv", index=False)
    pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "cash": [300000, 296440, 305380],
            "stock_equity": [1200000, 1100000, 1080000],
            "option_value": [0, 5000, 12000],
            "total_equity": [1500000, 1401440, 1397380],
            "daily_option_pnl": [0, 5000, 7000],
            "macro_state": ["NORMAL", "AI_SUPPLY_DISTORTION", "BULLWHIP_COLLAPSE"],
        }
    ).to_csv(report_dir / "equity_curve.csv", index=False)
    dd = 1397380 / 1500000 - 1
    pd.DataFrame({"max_drawdown": [dd]}).to_csv(report_dir / "summary.csv", index=False)
    pd.DataFrame({"item": ["Consumer sentiment", "AI server capex growth"], "value": [60, 80], "status": ["RED", "RED"]}).to_csv(report_dir / "macro_dashboard.csv", index=False)

    audit = run_report_audit(report_dir)

    assert (report_dir / "report_audit.csv").exists()
    assert (report_dir / "report_audit.md").exists()
    assert "status" in audit.columns
    assert (audit["check"] == "put_spread_open_paired").any()


def test_report_audit_flags_unbalanced_legs(tmp_path) -> None:
    report_dir = tmp_path
    pd.DataFrame(
        {
            "date": ["2024-01-02"],
            "position_id": ["PS-1"],
            "strategy": ["put_spread"],
            "action": ["BUY"],
            "cp": ["P"],
            "strike": [9000],
            "expiry": ["2024-03-01"],
            "quantity": [1],
            "price": [100],
            "cash_flow": [-5030],
            "cost": [30],
            "reason": ["open_put_spread"],
        }
    ).to_csv(report_dir / "trades.csv", index=False)
    pd.DataFrame({"date": ["2024-01-02"], "cash": [1], "stock_equity": [1], "option_value": [0], "total_equity": [2]}).to_csv(report_dir / "equity_curve.csv", index=False)
    pd.DataFrame({"max_drawdown": [0.0]}).to_csv(report_dir / "summary.csv", index=False)

    audit = run_report_audit(report_dir)

    failures = audit[audit["status"] == "FAIL"]["check"].tolist()
    assert "put_spread_open_paired" in failures
