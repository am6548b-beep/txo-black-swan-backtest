"""Diagnostics-only AI bullwhip risk data audit.

The first version audits whether the required scenario-aware indicators exist.
It does not create trades, fill missing data, or alter HedgeNeedScore.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


SUPPLY_DISTORTION_FIELDS = [
    "memory_revenue_yoy",
    "dram_price_proxy",
    "ddr5_price_proxy",
    "hbm_supply_pressure_proxy",
    "ai_server_revenue_yoy",
]

FALSE_DEMAND_BOOM_FIELDS = [
    "pcb_revenue_yoy",
    "mlcc_revenue_yoy",
    "driver_ic_revenue_yoy",
    "component_revenue_yoy",
    "component_revenue_growth_minus_end_demand_growth",
    "inventory_days_components",
    "accounts_receivable_growth",
]

DEMAND_BREAK_FIELDS = [
    "pc_shipments_yoy",
    "smartphone_shipments_yoy",
    "consumer_electronics_revenue_yoy",
    "real_wage_growth_yoy",
    "consumer_confidence",
    "retail_sales_yoy",
    "cpi_yoy",
    "policy_rate",
]

OPTIONAL_FIELDS = ["memory_maker_capex_mix"]

REQUIRED_FIELDS = ["date"] + SUPPLY_DISTORTION_FIELDS + FALSE_DEMAND_BOOM_FIELDS + DEMAND_BREAK_FIELDS

SUBSCORES = {
    "SupplyDistortionScore": SUPPLY_DISTORTION_FIELDS,
    "FalseDemandBoomScore": FALSE_DEMAND_BOOM_FIELDS,
    "DemandBreakRiskScore": DEMAND_BREAK_FIELDS,
}


def write_ai_bullwhip_diagnostics(data_dir: Path, report_dir: Path) -> pd.DataFrame:
    """Write AI bullwhip data availability audit reports."""

    report_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "ai_bullwhip_indicators.csv"
    rows = ai_bullwhip_audit_rows(path)
    audit = pd.DataFrame(rows)
    audit.to_csv(report_dir / "ai_bullwhip_risk_audit.csv", index=False)
    (report_dir / "ai_bullwhip_risk_audit.md").write_text(ai_bullwhip_audit_markdown(audit), encoding="utf-8")
    return audit


def ai_bullwhip_audit_rows(path: Path) -> list[dict[str, Any]]:
    """Return audit rows for an AI bullwhip indicator file path."""

    if not path.exists():
        return _missing_file_rows(path)
    try:
        data = pd.read_csv(path, parse_dates=["date"])
    except Exception as exc:  # pragma: no cover - defensive reporting path
        return [
            {
                "section": "file_status",
                "status": "FAIL",
                "detail": f"READ_ERROR: {exc}",
                "source_file": str(path),
                "AI_Bullwhip_Risk_Score_confidence": "MISSING",
            }
        ]
    if data.empty:
        rows = _missing_file_rows(path)
        rows[0]["detail"] = "ai_bullwhip_indicators.csv exists but has zero rows"
        return rows

    data = data.copy()
    available = [field for field in REQUIRED_FIELDS if field in data.columns]
    missing = [field for field in REQUIRED_FIELDS if field not in data.columns]
    rows: list[dict[str, Any]] = [
        {
            "section": "file_status",
            "status": "PASS" if not missing else "WARN",
            "source_file": str(path),
            "row_count": int(len(data)),
            "date_min": str(data["date"].min().date()) if "date" in data else "",
            "date_max": str(data["date"].max().date()) if "date" in data else "",
            "available_fields": ",".join(available),
            "missing_fields": ",".join(missing),
        }
    ]
    for field in REQUIRED_FIELDS:
        if field == "date":
            continue
        if field not in data:
            rows.append(
                {
                    "section": "field_coverage",
                    "field": field,
                    "status": "WARN",
                    "available": False,
                    "non_null_count": 0,
                    "coverage_ratio": 0.0,
                }
            )
            continue
        non_null = int(pd.to_numeric(data[field], errors="coerce").notna().sum())
        ratio = non_null / len(data) if len(data) else 0.0
        rows.append(
            {
                "section": "field_coverage",
                "field": field,
                "status": "PASS" if ratio > 0 else "WARN",
                "available": True,
                "non_null_count": non_null,
                "coverage_ratio": ratio,
            }
        )
    subscore_coverages: dict[str, float] = {}
    for subscore, fields in SUBSCORES.items():
        field_ratios = [_field_coverage(data, field) for field in fields]
        coverage = sum(field_ratios) / len(fields)
        subscore_coverages[subscore] = coverage
        rows.append(
            {
                "section": "subscore_coverage",
                "subscore": subscore,
                "status": "PASS" if coverage >= 0.60 else ("WARN" if coverage > 0 else "FAIL"),
                "can_compute_subscore": coverage >= 0.60,
                "coverage_ratio": coverage,
                "available_fields": ",".join(field for field in fields if field in data.columns and _field_coverage(data, field) > 0),
                "missing_fields": ",".join(field for field in fields if field not in data.columns or _field_coverage(data, field) == 0),
            }
        )
    overall = sum(subscore_coverages.values()) / len(subscore_coverages)
    confidence = confidence_from_coverage(overall, list(subscore_coverages.values()))
    rows.append(
        {
            "section": "score_confidence",
            "status": "PASS" if confidence != "MISSING" else "FAIL",
            "proxy_coverage_score": overall,
            "AI_Bullwhip_Risk_Score_confidence": confidence,
            "computed_subscores": ",".join(name for name, value in subscore_coverages.items() if value >= 0.60),
            "missing_subscores": ",".join(name for name, value in subscore_coverages.items() if value < 0.60),
            "detail": "Scenario-aware diagnostic only; not a backtest-optimized signal.",
        }
    )
    return rows


def confidence_from_coverage(overall_coverage: float, subscore_coverages: list[float]) -> str:
    """Map proxy coverage to confidence without using performance outcomes."""

    if overall_coverage <= 0:
        return "MISSING"
    if overall_coverage >= 0.80 and all(value >= 0.60 for value in subscore_coverages):
        return "HIGH"
    if overall_coverage >= 0.40 and sum(value >= 0.60 for value in subscore_coverages) >= 2:
        return "MEDIUM"
    return "LOW"


def ai_bullwhip_audit_markdown(audit: pd.DataFrame) -> str:
    """Render a concise markdown audit without ranking language."""

    lines = [
        "# AI Bullwhip Risk Audit",
        "",
        "Diagnostics-only audit for AI bullwhip indicator availability.",
        "",
        "This report does not create trades, alter HedgeNeedScore, fill missing data, or tune weights.",
        "",
    ]
    if audit.empty:
        lines.append("- status: FAIL")
        lines.append("- detail: audit produced no rows")
    else:
        file_rows = audit[audit["section"].eq("file_status")]
        if not file_rows.empty:
            row = file_rows.iloc[0]
            lines.append("## File Status")
            lines.append(f"- status: {row.get('status', '')}")
            lines.append(f"- row_count: {row.get('row_count', '')}")
            lines.append(f"- date_range: {row.get('date_min', '')} to {row.get('date_max', '')}")
            lines.append(f"- available_fields: {row.get('available_fields', '')}")
            lines.append(f"- missing_fields: {row.get('missing_fields', '')}")
            lines.append("")
        subscore = audit[audit["section"].eq("subscore_coverage")]
        if not subscore.empty:
            lines.append("## Subscore Coverage")
            for _, row in subscore.iterrows():
                lines.append(
                    f"- {row.get('subscore', '')}: status={row.get('status', '')}, "
                    f"coverage={row.get('coverage_ratio', '')}, can_compute={row.get('can_compute_subscore', '')}"
                )
            lines.append("")
        confidence = audit[audit["section"].eq("score_confidence")]
        if not confidence.empty:
            row = confidence.iloc[0]
            lines.append("## Score Confidence")
            lines.append(f"- proxy_coverage_score: {row.get('proxy_coverage_score', '')}")
            lines.append(f"- confidence: {row.get('AI_Bullwhip_Risk_Score_confidence', '')}")
            lines.append(f"- computed_subscores: {row.get('computed_subscores', '')}")
            lines.append(f"- missing_subscores: {row.get('missing_subscores', '')}")
            lines.append("")
    lines.extend(
        [
            "## Warning",
            "- This is a scenario-aware diagnostic, not a backtest-optimized signal.",
            "- Missing data remains missing; no placeholder values are inserted.",
            "- The module does not create trades.",
        ]
    )
    text = "\n".join(lines) + "\n"
    lowered = text.lower()
    if "best" in lowered or "recommend" in lowered:
        raise ValueError("AI bullwhip audit report contains disallowed wording")
    return text


def _missing_file_rows(path: Path) -> list[dict[str, Any]]:
    missing_fields = [field for field in REQUIRED_FIELDS if field != "date"]
    rows: list[dict[str, Any]] = [
        {
            "section": "file_status",
            "status": "WARN",
            "source_file": str(path),
            "detail": "missing ai_bullwhip_indicators.csv",
            "available_fields": "",
            "missing_fields": ",".join(REQUIRED_FIELDS),
            "proxy_coverage_score": 0.0,
            "AI_Bullwhip_Risk_Score_confidence": "MISSING",
        }
    ]
    for subscore in SUBSCORES:
        rows.append(
            {
                "section": "subscore_coverage",
                "subscore": subscore,
                "status": "FAIL",
                "can_compute_subscore": False,
                "coverage_ratio": 0.0,
                "available_fields": "",
                "missing_fields": ",".join(SUBSCORES[subscore]),
            }
        )
    rows.append(
        {
            "section": "score_confidence",
            "status": "FAIL",
            "proxy_coverage_score": 0.0,
            "AI_Bullwhip_Risk_Score_confidence": "MISSING",
            "computed_subscores": "",
            "missing_subscores": ",".join(SUBSCORES),
            "detail": f"missing fields: {','.join(missing_fields)}",
        }
    )
    return rows


def _field_coverage(data: pd.DataFrame, field: str) -> float:
    if field not in data or data.empty:
        return 0.0
    return float(pd.to_numeric(data[field], errors="coerce").notna().sum() / len(data))
