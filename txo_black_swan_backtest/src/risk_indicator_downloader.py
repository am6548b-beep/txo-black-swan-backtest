"""Download official or traceable risk indicators.

The downloader is deliberately conservative: failed sources are audited and
missing fields remain NaN. No mock values, proxies, or synthetic fills are
written into risk_indicators.csv.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


RISK_INDICATOR_COLUMNS = [
    "date",
    "tw_vix",
    "tw_vix_is_proxy",
    "put_call_volume_ratio",
    "put_call_oi_ratio",
    "twse_pe",
    "twse_pb",
    "twse_dividend_yield",
    "foreign_futures_net_position",
    "usdtwd",
    "dxy",
    "us10y",
    "us2y",
    "cpi_yoy",
    "core_cpi_yoy",
    "source_coverage_score",
]


@dataclass(frozen=True)
class RiskSource:
    source_name: str
    provider: str
    url: str
    api_endpoint: str
    license_url: str
    frequency: str
    fields: tuple[str, ...]
    status: str
    notes: str


def read_source_registry(path: Path) -> list[RiskSource]:
    """Read the small project-owned YAML registry without requiring PyYAML."""

    if not path.exists():
        return []
    sources: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    current_list_key: str | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped == "sources:":
            continue
        if stripped.startswith("- source_name:"):
            if current:
                sources.append(current)
            current = {"source_name": _yaml_value(stripped.split(":", 1)[1])}
            current_list_key = None
            continue
        if current is None:
            continue
        if stripped.endswith(":") and not stripped.startswith("- "):
            current_list_key = stripped[:-1]
            current[current_list_key] = []
            continue
        if stripped.startswith("- ") and current_list_key:
            current[current_list_key].append(_yaml_value(stripped[2:]))
            continue
        if ":" in stripped:
            key, value = stripped.split(":", 1)
            current[key] = _yaml_value(value)
            current_list_key = None
    if current:
        sources.append(current)
    out = []
    for item in sources:
        out.append(
            RiskSource(
                source_name=str(item.get("source_name", "")),
                provider=str(item.get("provider", "")),
                url=str(item.get("url", "")),
                api_endpoint=str(item.get("api_endpoint", "")),
                license_url=str(item.get("license_url", "")),
                frequency=str(item.get("frequency", "")),
                fields=tuple(item.get("fields", []) or []),
                status=str(item.get("status", "")),
                notes=str(item.get("notes", "")),
            )
        )
    return out


def download_risk_indicators(
    start_date: str,
    end_date: str,
    out_dir: Path,
    processed_dir: Path,
    report_dir: Path,
    registry_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    sources = read_source_registry(registry_path)
    frames: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    for source in sources:
        frame, audit = _download_source(source, start_date, end_date, out_dir)
        frames.append(frame)
        audit_rows.append(audit)
    risk = _combine_frames(frames, start_date, end_date)
    risk.to_csv(processed_dir / "risk_indicators.csv", index=False)
    audit = pd.DataFrame(audit_rows)
    if not audit.empty:
        audit["failed_source"] = audit["status"].astype(str) == "failed"
    audit.to_csv(report_dir / "risk_indicator_download_audit.csv", index=False)
    (report_dir / "risk_indicator_download_audit.md").write_text(_audit_markdown(audit), encoding="utf-8")
    return risk, audit


def _download_source(source: RiskSource, start_date: str, end_date: str, out_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not source.api_endpoint:
        return pd.DataFrame(), _audit(source, "planned", "empty endpoint; not implemented", pd.DataFrame())
    try:
        data = _fetch_json(source.api_endpoint)
        raw_path = out_dir / f"{source.source_name}.json"
        raw_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        raw = pd.DataFrame(data if isinstance(data, list) else data.get("data", []))
        frame = _normalize_source(source, raw)
        frame = _filter_dates(frame, start_date, end_date)
        status = "implemented" if not frame.empty else "failed"
        detail = "" if not frame.empty else "downloaded but no normalizable rows"
        return frame, _audit(source, status, detail, frame)
    except Exception as exc:  # noqa: BLE001 - audit source failures without crashing.
        return pd.DataFrame(), _audit(source, "failed", f"{type(exc).__name__}: {exc}", pd.DataFrame())


def _normalize_source(source: RiskSource, raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    if source.source_name == "taifex_put_call_ratio":
        return _normalize_put_call_ratio(raw)
    if source.source_name == "taifex_txo_volatility_index":
        return _normalize_tw_vix(raw)
    if source.source_name == "twse_valuation_by_date":
        return _normalize_twse_valuation(raw)
    if source.source_name == "taifex_daily_foreign_exchange_rates":
        return _normalize_usdtwd(raw)
    if source.source_name == "taifex_major_institutional_traders_futures_options":
        return _normalize_foreign_position(raw)
    return pd.DataFrame()


def _normalize_put_call_ratio(raw: pd.DataFrame) -> pd.DataFrame:
    cols = {str(col): str(col) for col in raw.columns}
    date_col = _find_col(cols, ["date", "日期", "交易日期"])
    vol_col = _find_col(cols, ["put/call volume", "成交量比率", "買賣權成交量比率", "putcallratio"])
    oi_col = _find_col(cols, ["put/call open interest", "未沖銷契約量比率", "買賣權未沖銷契約量比率"])
    if not date_col:
        return pd.DataFrame()
    out = pd.DataFrame({"date": _parse_dates(raw[date_col])})
    out["put_call_volume_ratio"] = _numeric(raw[vol_col]) if vol_col else np.nan
    out["put_call_oi_ratio"] = _numeric(raw[oi_col]) if oi_col else np.nan
    out["put_call_source"] = "TAIFEX"
    return out.dropna(subset=["date"])


def _normalize_tw_vix(raw: pd.DataFrame) -> pd.DataFrame:
    cols = {str(col): str(col) for col in raw.columns}
    date_col = _find_col(cols, ["date", "日期", "交易日期"])
    vix_col = _find_col(cols, ["vix", "波動率", "volatility"])
    if not date_col or not vix_col:
        return pd.DataFrame()
    out = pd.DataFrame({"date": _parse_dates(raw[date_col]), "tw_vix": _numeric(raw[vix_col])})
    out["tw_vix_source"] = "TAIFEX"
    out["tw_vix_is_proxy"] = False
    return out.dropna(subset=["date"])


def _normalize_twse_valuation(raw: pd.DataFrame) -> pd.DataFrame:
    cols = {str(col): str(col) for col in raw.columns}
    date_col = _find_col(cols, ["date", "日期"])
    pe_col = _find_col(cols, ["pe", "本益比"])
    pb_col = _find_col(cols, ["pb", "股價淨值比"])
    div_col = _find_col(cols, ["dividend", "殖利率"])
    if not date_col:
        return pd.DataFrame()
    frame = pd.DataFrame({"date": _parse_dates(raw[date_col])})
    frame["twse_pe"] = _numeric(raw[pe_col]) if pe_col else np.nan
    frame["twse_pb"] = _numeric(raw[pb_col]) if pb_col else np.nan
    frame["twse_dividend_yield"] = _numeric(raw[div_col]) if div_col else np.nan
    frame = frame.dropna(subset=["date"])
    if frame.empty:
        return frame
    out = frame.groupby("date", as_index=False)[["twse_pe", "twse_pb", "twse_dividend_yield"]].median(numeric_only=True)
    out["twse_valuation_source"] = "TWSE"
    return out


def _normalize_usdtwd(raw: pd.DataFrame) -> pd.DataFrame:
    cols = {str(col): str(col) for col in raw.columns}
    date_col = _find_col(cols, ["date", "日期"])
    usd_col = _find_col(cols, ["usd", "美元", "usdtwd"])
    if not date_col or not usd_col:
        return pd.DataFrame()
    return pd.DataFrame({"date": _parse_dates(raw[date_col]), "usdtwd": _numeric(raw[usd_col])}).dropna(subset=["date"])


def _normalize_foreign_position(raw: pd.DataFrame) -> pd.DataFrame:
    cols = {str(col): str(col) for col in raw.columns}
    date_col = _find_col(cols, ["date", "日期", "交易日期"])
    net_col = _find_col(cols, ["外資", "net", "未平倉餘額"])
    if not date_col or not net_col:
        return pd.DataFrame()
    return pd.DataFrame({"date": _parse_dates(raw[date_col]), "foreign_futures_net_position": _numeric(raw[net_col])}).dropna(subset=["date"])


def _combine_frames(frames: list[pd.DataFrame], start_date: str, end_date: str) -> pd.DataFrame:
    non_empty = [frame.copy() for frame in frames if not frame.empty]
    if not non_empty:
        return pd.DataFrame(columns=RISK_INDICATOR_COLUMNS)
    out = non_empty[0]
    for frame in non_empty[1:]:
        out = out.merge(frame, on="date", how="outer")
    out = _filter_dates(out, start_date, end_date).sort_values("date")
    for col in RISK_INDICATOR_COLUMNS:
        if col not in out.columns:
            out[col] = False if col == "tw_vix_is_proxy" else np.nan
    coverage_cols = [col for col in RISK_INDICATOR_COLUMNS if col not in {"date", "source_coverage_score", "tw_vix_is_proxy"}]
    out["source_coverage_score"] = out[coverage_cols].notna().mean(axis=1)
    return out[RISK_INDICATOR_COLUMNS]


def _audit(source: RiskSource, status: str, detail: str, frame: pd.DataFrame) -> dict[str, Any]:
    duplicate_dates = int(frame["date"].duplicated().sum()) if not frame.empty and "date" in frame else 0
    gap_count = _gap_count(frame["date"]) if not frame.empty and "date" in frame else 0
    return {
        "source_name": source.source_name,
        "provider": source.provider,
        "status": status,
        "row_count": int(len(frame)),
        "date_min": str(frame["date"].min().date()) if not frame.empty and "date" in frame else "",
        "date_max": str(frame["date"].max().date()) if not frame.empty and "date" in frame else "",
        "missing_ratio": float(frame.drop(columns=["date"], errors="ignore").isna().mean().mean()) if not frame.empty else "",
        "duplicate_date_count": duplicate_dates,
        "gap_gt_7d_count": gap_count,
        "field_types": json.dumps({col: str(dtype) for col, dtype in frame.dtypes.items()}, ensure_ascii=False) if not frame.empty else "",
        "source_url": source.api_endpoint or source.url,
        "license_url": source.license_url,
        "has_proxy": False,
        "detail": detail,
    }


def _audit_markdown(audit: pd.DataFrame) -> str:
    lines = [
        "# Risk Indicator Download Audit",
        "",
        "Only official or traceable sources are attempted. Failed sources leave fields empty; no mock values are filled.",
        "",
        "## Sources",
        "",
    ]
    if audit.empty:
        lines.append("- No sources configured.")
    else:
        for row in audit.itertuples(index=False):
            lines.append(f"- {row.source_name}: {row.status}, rows={row.row_count}, detail={row.detail}")
    lines.extend(
        [
            "",
            "## Required Limitations",
            "",
            "- Missing fields remain NaN.",
            "- Proxy values are not written as official data.",
            "- No trading or backtest interpretation is performed.",
        ]
    )
    return "\n".join(lines) + "\n"


def _fetch_json(url: str) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": "txo-risk-indicator-downloader/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        raw = response.read().decode("utf-8-sig")
    return json.loads(raw)


def _filter_dates(frame: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    if frame.empty or "date" not in frame:
        return frame
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out[(out["date"] >= pd.Timestamp(start_date)) & (out["date"] <= pd.Timestamp(end_date))].dropna(subset=["date"])


def _parse_dates(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.strip()
    parsed = pd.to_datetime(text, errors="coerce")
    roc = parsed.isna() & text.str.match(r"^\d{2,3}/\d{1,2}/\d{1,2}$", na=False)
    if roc.any():
        def parse_roc(value: str) -> pd.Timestamp:
            year, month, day = [int(part) for part in value.split("/")]
            return pd.Timestamp(year + 1911, month, day)
        parsed.loc[roc] = text.loc[roc].map(parse_roc)
    compact = parsed.isna() & text.str.match(r"^\d{8}$", na=False)
    if compact.any():
        parsed.loc[compact] = pd.to_datetime(text.loc[compact], format="%Y%m%d", errors="coerce")
    return parsed


def _numeric(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values.astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False), errors="coerce")


def _find_col(cols: dict[str, str], keywords: list[str]) -> str | None:
    lowered = {key: key.lower().replace(" ", "") for key in cols}
    for keyword in keywords:
        pattern = keyword.lower().replace(" ", "")
        for original, lower in lowered.items():
            if pattern in lower:
                return original
    return None


def _gap_count(dates: pd.Series) -> int:
    clean = pd.to_datetime(dates, errors="coerce").dropna().sort_values()
    if len(clean) < 2:
        return 0
    return int((clean.diff().dt.days > 7).sum())


def _yaml_value(value: str) -> str:
    return value.strip().strip('"').strip("'")
