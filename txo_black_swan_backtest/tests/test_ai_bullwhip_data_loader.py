from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.ai_bullwhip_data_loader import _audit_markdown, build_ai_bullwhip_indicators


def test_missing_raw_folder_does_not_crash(tmp_path: Path) -> None:
    processed, audit = build_ai_bullwhip_indicators(tmp_path / "missing", tmp_path / "processed", tmp_path / "reports")

    assert processed.empty
    assert not audit.empty
    assert (tmp_path / "processed" / "ai_bullwhip_indicators.csv").exists()


def test_monthly_data_can_normalize(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    pd.DataFrame(
        {
            "date": ["2024-01-31", "2024-02-29"],
            "source_frequency": ["MONTHLY", "MONTHLY"],
            "memory_revenue_yoy": [0.1, 0.2],
            "pcb_revenue_yoy": [0.3, 0.4],
        }
    ).to_csv(raw / "monthly.csv", index=False)

    processed, _ = build_ai_bullwhip_indicators(raw, tmp_path / "processed", tmp_path / "reports")

    assert len(processed) == 2
    assert set(processed["source_frequency"]) == {"MONTHLY"}
    assert processed["memory_revenue_yoy"].tolist() == [0.1, 0.2]


def test_quarterly_data_can_normalize(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    pd.DataFrame(
        {
            "date": ["2024-03-31", "2024-06-30"],
            "source_frequency": ["QUARTERLY", "QUARTERLY"],
            "inventory_days_components": [80, 95],
            "accounts_receivable_growth": [0.05, 0.08],
        }
    ).to_csv(raw / "quarterly.csv", index=False)

    processed, _ = build_ai_bullwhip_indicators(raw, tmp_path / "processed", tmp_path / "reports")

    assert len(processed) == 2
    assert set(processed["source_frequency"]) == {"QUARTERLY"}
    assert processed["inventory_days_components"].tolist() == [80, 95]


def test_missing_fields_remain_nan(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    pd.DataFrame({"date": ["2024-01-31"], "memory_revenue_yoy": [0.1]}).to_csv(raw / "partial.csv", index=False)

    processed, _ = build_ai_bullwhip_indicators(raw, tmp_path / "processed", tmp_path / "reports")

    assert pd.isna(processed.iloc[0]["pc_shipments_yoy"])
    assert pd.isna(processed.iloc[0]["policy_rate"])


def test_forward_filled_is_explicitly_marked(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    pd.DataFrame(
        {
            "date": ["2024-01-31"],
            "memory_revenue_yoy": [0.1],
            "forward_filled": [True],
            "data_lag_days": [45],
            "latest_available_date": ["2023-12-15"],
        }
    ).to_csv(raw / "manual.csv", index=False)

    processed, _ = build_ai_bullwhip_indicators(raw, tmp_path / "processed", tmp_path / "reports")

    assert bool(processed.iloc[0]["forward_filled"]) is True
    assert processed.iloc[0]["data_lag_days"] == 45
    assert processed.iloc[0]["latest_available_date"] == "2023-12-15"


def test_ai_bullwhip_data_build_report_has_no_disallowed_wording(tmp_path: Path) -> None:
    _, audit = build_ai_bullwhip_indicators(tmp_path / "missing", tmp_path / "processed", tmp_path / "reports")
    text = _audit_markdown(audit).lower()

    assert "best" not in text
    assert "recommend" not in text


def test_build_script_outputs_do_not_create_trades_or_modify_hedge_need(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    reports = tmp_path / "reports"
    processed_dir = tmp_path / "processed"
    raw.mkdir()
    reports.mkdir()
    processed_dir.mkdir()
    (reports / "hedge_need_score.csv").write_text("date,HedgeNeedScore\n2024-01-31,50\n", encoding="utf-8")
    before = (reports / "hedge_need_score.csv").read_text(encoding="utf-8")
    pd.DataFrame({"date": ["2024-01-31"], "memory_revenue_yoy": [0.1]}).to_csv(raw / "manual.csv", index=False)

    build_ai_bullwhip_indicators(raw, processed_dir, reports)

    assert (reports / "hedge_need_score.csv").read_text(encoding="utf-8") == before
    assert not (reports / "trades.csv").exists()
    assert not (reports / "risk_scaled_hedge_trades.csv").exists()
