"""Manual AI bullwhip indicator ingestion scaffold.

The builder normalizes manually maintained monthly / quarterly CSV files into
data/processed/ai_bullwhip_indicators.csv. It does not create scores, trades,
or HedgeNeedScore inputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .ai_bullwhip_risk import DEMAND_BREAK_FIELDS, FALSE_DEMAND_BOOM_FIELDS, REQUIRED_FIELDS, SUBSCORES, SUPPLY_DISTORTION_FIELDS


META_COLUMNS = ["source_frequency", "forward_filled", "data_lag_days", "latest_available_date", "source_file"]
OUTPUT_COLUMNS = REQUIRED_FIELDS + META_COLUMNS


def build_ai_bullwhip_indicators(raw_dir: Path, processed_dir: Path, report_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build processed AI bullwhip indicators from manual raw CSV files."""

    processed_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    raw_files = sorted(raw_dir.glob("*.csv")) if raw_dir.exists() else []
    if not raw_files:
        processed = _empty_processed()
        audit = pd.DataFrame(_missing_raw_audit(raw_dir))
        processed.to_csv(processed_dir / "ai_bullwhip_indicators.csv", index=False)
        audit.to_csv(report_dir / "ai_bullwhip_data_build_audit.csv", index=False)
        (report_dir / "ai_bullwhip_data_build_audit.md").write_text(_audit_markdown(audit), encoding="utf-8")
        return processed, audit

    frames: list[pd.DataFrame] = []
    file_rows: list[dict[str, Any]] = []
    for path in raw_files:
        normalized, row = _read_manual_file(path)
        file_rows.append(row)
        if not normalized.empty:
            frames.append(normalized)
    processed = _combine_frames(frames)
    processed.to_csv(processed_dir / "ai_bullwhip_indicators.csv", index=False)
    audit = pd.DataFrame(file_rows + _processed_audit_rows(processed, raw_files))
    audit.to_csv(report_dir / "ai_bullwhip_data_build_audit.csv", index=False)
    (report_dir / "ai_bullwhip_data_build_audit.md").write_text(_audit_markdown(audit), encoding="utf-8")
    return processed, audit


def _read_manual_file(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    try:
        data = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - defensive audit path
        return _empty_processed(), {"section": "raw_file", "source_file": str(path), "status": "FAIL", "detail": f"READ_ERROR: {exc}"}
    if data.empty:
        return _empty_processed(), {"section": "raw_file", "source_file": str(path), "status": "WARN", "raw_rows": 0, "detail": "EMPTY_FILE"}
    missing_required = ["date"] if "date" not in data.columns else []
    if missing_required:
        return _empty_processed(), {"section": "raw_file", "source_file": str(path), "status": "FAIL", "raw_rows": len(data), "missing_required_columns": ",".join(missing_required)}
    out = _empty_processed()
    out = pd.DataFrame(index=data.index)
    out["date"] = pd.to_datetime(data["date"], errors="coerce")
    for field in REQUIRED_FIELDS:
        if field == "date":
            continue
        out[field] = pd.to_numeric(data[field], errors="coerce") if field in data.columns else pd.NA
    source_frequency = _source_frequency(data, out)
    out["source_frequency"] = source_frequency
    out["forward_filled"] = _bool_series(data.get("forward_filled", False), len(out))
    out["data_lag_days"] = pd.to_numeric(data.get("data_lag_days", pd.Series([pd.NA] * len(out))), errors="coerce")
    latest = pd.to_datetime(data.get("latest_available_date", out["date"]), errors="coerce")
    out["latest_available_date"] = latest.dt.strftime("%Y-%m-%d")
    out["source_file"] = path.name
    out = out[OUTPUT_COLUMNS].sort_values("date").reset_index(drop=True)
    invalid_dates = int(out["date"].isna().sum())
    valid = out.dropna(subset=["date"]).copy()
    valid["date"] = valid["date"].dt.strftime("%Y-%m-%d")
    row = {
        "section": "raw_file",
        "source_file": str(path),
        "status": "PASS" if invalid_dates == 0 else "WARN",
        "raw_rows": int(len(data)),
        "normalized_rows": int(len(valid)),
        "date_min": valid["date"].min() if not valid.empty else "",
        "date_max": valid["date"].max() if not valid.empty else "",
        "source_frequency": source_frequency,
        "invalid_date_rows": invalid_dates,
        "missing_fields": ",".join(field for field in REQUIRED_FIELDS if field not in data.columns),
    }
    return valid, row


def _combine_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return _empty_processed()
    combined = pd.concat(frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
    combined = combined.sort_values(["date", "source_file"]).reset_index(drop=True)
    # Last non-null manual value for a duplicated date/field wins, preserving metadata from the last row.
    rows: list[dict[str, Any]] = []
    for date, group in combined.groupby("date", sort=True):
        row: dict[str, Any] = {"date": date.strftime("%Y-%m-%d")}
        for field in REQUIRED_FIELDS:
            if field == "date":
                continue
            values = pd.to_numeric(group[field], errors="coerce").dropna()
            row[field] = values.iloc[-1] if not values.empty else pd.NA
        for field in META_COLUMNS:
            value = group[field].dropna().iloc[-1] if field in group and not group[field].dropna().empty else pd.NA
            row[field] = value
        rows.append(row)
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def _processed_audit_rows(processed: pd.DataFrame, raw_files: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {
            "section": "processed_summary",
            "status": "PASS" if not processed.empty else "WARN",
            "raw_files_count": int(len(raw_files)),
            "processed_rows": int(len(processed)),
            "date_min": processed["date"].min() if not processed.empty else "",
            "date_max": processed["date"].max() if not processed.empty else "",
        }
    ]
    for field in REQUIRED_FIELDS:
        if field == "date":
            continue
        ratio = _missing_ratio(processed, field)
        rows.append({"section": "missing_ratio_per_field", "field": field, "status": "PASS" if ratio < 1.0 else "WARN", "missing_ratio": ratio})
    for subscore, fields in SUBSCORES.items():
        coverage = sum(1.0 - _missing_ratio(processed, field) for field in fields) / len(fields) if not processed.empty else 0.0
        rows.append(
            {
                "section": "field_coverage_by_subscore",
                "subscore": subscore,
                "coverage_ratio": coverage,
                "can_compute_subscore": coverage >= 0.60,
                "status": "PASS" if coverage >= 0.60 else ("WARN" if coverage > 0 else "FAIL"),
            }
        )
    if not processed.empty:
        for frequency, count in processed["source_frequency"].fillna("UNKNOWN").astype(str).value_counts().items():
            rows.append({"section": "frequency_distribution", "source_frequency": frequency, "count": int(count)})
        lag = pd.to_numeric(processed["data_lag_days"], errors="coerce")
        rows.append(
            {
                "section": "data_lag_summary",
                "min_data_lag_days": float(lag.min()) if lag.notna().any() else "",
                "median_data_lag_days": float(lag.median()) if lag.notna().any() else "",
                "max_data_lag_days": float(lag.max()) if lag.notna().any() else "",
            }
        )
        confidence = _confidence_distribution(processed)
        for label, count in confidence.items():
            rows.append({"section": "confidence_distribution", "confidence": label, "count": int(count)})
    return rows


def _source_frequency(raw: pd.DataFrame, normalized: pd.DataFrame) -> str:
    if "source_frequency" in raw.columns and raw["source_frequency"].notna().any():
        value = str(raw["source_frequency"].dropna().iloc[0]).upper()
        if value in {"MONTHLY", "QUARTERLY"}:
            return value
    dates = pd.to_datetime(normalized["date"], errors="coerce").dropna().sort_values()
    if len(dates) < 2:
        return "MONTHLY"
    median_days = dates.diff().dropna().dt.days.median()
    return "QUARTERLY" if median_days >= 70 else "MONTHLY"


def _bool_series(value: Any, length: int) -> pd.Series:
    if isinstance(value, pd.Series):
        return value.fillna(False).map(lambda item: str(item).strip().lower() in {"1", "true", "yes", "y"})
    return pd.Series([str(value).strip().lower() in {"1", "true", "yes", "y"}] * length)


def _missing_ratio(processed: pd.DataFrame, field: str) -> float:
    if processed.empty or field not in processed:
        return 1.0
    return float(pd.to_numeric(processed[field], errors="coerce").isna().mean())


def _confidence_distribution(processed: pd.DataFrame) -> dict[str, int]:
    if processed.empty:
        return {"MISSING": 0}
    rows: dict[str, int] = {"HIGH": 0, "MEDIUM": 0, "LOW": 0, "MISSING": 0}
    for _, row in processed.iterrows():
        coverages = []
        for fields in SUBSCORES.values():
            coverages.append(sum(pd.notna(pd.to_numeric(row.get(field), errors="coerce")) for field in fields) / len(fields))
        overall = sum(coverages) / len(coverages)
        label = "MISSING" if overall <= 0 else ("HIGH" if overall >= 0.80 and all(value >= 0.60 for value in coverages) else ("MEDIUM" if overall >= 0.40 and sum(value >= 0.60 for value in coverages) >= 2 else "LOW"))
        rows[label] += 1
    return {key: value for key, value in rows.items() if value > 0}


def _missing_raw_audit(raw_dir: Path) -> list[dict[str, Any]]:
    rows = [
        {
            "section": "processed_summary",
            "status": "WARN",
            "raw_files_count": 0,
            "processed_rows": 0,
            "detail": f"missing or empty raw folder: {raw_dir}",
        }
    ]
    for subscore in SUBSCORES:
        rows.append({"section": "field_coverage_by_subscore", "subscore": subscore, "coverage_ratio": 0.0, "can_compute_subscore": False, "status": "FAIL"})
    return rows


def _empty_processed() -> pd.DataFrame:
    return pd.DataFrame(columns=OUTPUT_COLUMNS)


def _audit_markdown(audit: pd.DataFrame) -> str:
    lines = [
        "# AI Bullwhip Data Build Audit",
        "",
        "Diagnostics-only audit for manual AI bullwhip data ingestion.",
        "",
        "This builder does not create scores, trades, or HedgeNeedScore inputs.",
        "",
    ]
    if audit.empty:
        lines.append("- status: WARN")
        lines.append("- detail: no audit rows")
    else:
        summary = audit[audit["section"].eq("processed_summary")]
        if not summary.empty:
            row = summary.iloc[0]
            lines.append("## Processed Summary")
            lines.append(f"- raw_files_count: {row.get('raw_files_count', '')}")
            lines.append(f"- processed_rows: {row.get('processed_rows', '')}")
            lines.append(f"- date_range: {row.get('date_min', '')} to {row.get('date_max', '')}")
            lines.append("")
        subscore = audit[audit["section"].eq("field_coverage_by_subscore")]
        if not subscore.empty:
            lines.append("## Subscore Coverage")
            for _, row in subscore.iterrows():
                lines.append(f"- {row.get('subscore')}: coverage={row.get('coverage_ratio')}, can_compute={row.get('can_compute_subscore')}")
            lines.append("")
        freq = audit[audit["section"].eq("frequency_distribution")]
        if not freq.empty:
            lines.append("## Frequency Distribution")
            for _, row in freq.iterrows():
                lines.append(f"- {row.get('source_frequency')}: {row.get('count')}")
            lines.append("")
    lines.extend(
        [
            "## Guardrails",
            "- Monthly and quarterly rows remain at their source frequency.",
            "- Missing fields remain NaN.",
            "- No placeholder values are inserted.",
        ]
    )
    text = "\n".join(lines) + "\n"
    lowered = text.lower()
    if "best" in lowered or "recommend" in lowered:
        raise ValueError("AI bullwhip data build audit contains disallowed wording")
    return text
