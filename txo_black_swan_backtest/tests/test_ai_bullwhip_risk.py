from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.ai_bullwhip_risk import (
    REQUIRED_FIELDS,
    ai_bullwhip_audit_markdown,
    ai_bullwhip_audit_rows,
    confidence_from_coverage,
    write_ai_bullwhip_diagnostics,
)


def test_missing_ai_bullwhip_file_does_not_crash(tmp_path: Path) -> None:
    audit = write_ai_bullwhip_diagnostics(tmp_path, tmp_path / "reports")

    assert not audit.empty
    assert "MISSING" in set(audit["AI_Bullwhip_Risk_Score_confidence"].dropna())
    assert (tmp_path / "reports" / "ai_bullwhip_risk_audit.csv").exists()


def test_missing_fields_are_audited(tmp_path: Path) -> None:
    path = tmp_path / "ai_bullwhip_indicators.csv"
    pd.DataFrame({"date": ["2024-01-31"], "memory_revenue_yoy": [0.2]}).to_csv(path, index=False)

    audit = pd.DataFrame(ai_bullwhip_audit_rows(path))
    file_row = audit[audit["section"].eq("file_status")].iloc[0]

    assert file_row["status"] == "WARN"
    assert "pc_shipments_yoy" in file_row["missing_fields"]
    assert audit[audit["section"].eq("field_coverage")]["field"].isin(["memory_revenue_yoy"]).any()


def test_score_confidence_changes_with_data_coverage() -> None:
    assert confidence_from_coverage(0.0, [0.0, 0.0, 0.0]) == "MISSING"
    assert confidence_from_coverage(0.20, [0.60, 0.0, 0.0]) == "LOW"
    assert confidence_from_coverage(0.50, [0.70, 0.65, 0.15]) == "MEDIUM"
    assert confidence_from_coverage(0.85, [0.80, 0.90, 0.75]) == "HIGH"


def test_full_fields_can_compute_all_subscores(tmp_path: Path) -> None:
    path = tmp_path / "ai_bullwhip_indicators.csv"
    data = {field: [1.0] for field in REQUIRED_FIELDS if field != "date"}
    data["date"] = ["2024-01-31"]
    pd.DataFrame(data).to_csv(path, index=False)

    audit = pd.DataFrame(ai_bullwhip_audit_rows(path))
    subscores = audit[audit["section"].eq("subscore_coverage")]

    assert set(subscores["can_compute_subscore"]) == {True}
    assert audit[audit["section"].eq("score_confidence")].iloc[0]["AI_Bullwhip_Risk_Score_confidence"] == "HIGH"


def test_ai_bullwhip_report_has_no_disallowed_wording(tmp_path: Path) -> None:
    audit = write_ai_bullwhip_diagnostics(tmp_path, tmp_path / "reports")
    text = ai_bullwhip_audit_markdown(audit).lower()

    assert "best" not in text
    assert "recommend" not in text


def test_ai_bullwhip_module_does_not_generate_trades(tmp_path: Path) -> None:
    report_dir = tmp_path / "reports"
    write_ai_bullwhip_diagnostics(tmp_path, report_dir)

    assert not (report_dir / "trades.csv").exists()
    assert not (report_dir / "risk_scaled_hedge_trades.csv").exists()
