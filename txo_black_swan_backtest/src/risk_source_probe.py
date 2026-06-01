"""Probe official risk indicator endpoints without writing processed data."""

from __future__ import annotations

import csv
import io
import json
import ssl
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

from .risk_indicator_downloader import read_source_registry


SWAGGER_URLS = [
    ("taifex_swagger", "https://openapi.taifex.com.tw/swagger.json"),
    ("twse_swagger", "https://openapi.twse.com.tw/v1/swagger.json"),
]

KNOWN_ENDPOINTS = [
    ("taifex_pc_ratio_excel", "https://www.taifex.com.tw/cht/3/pcRatioExcel"),
    ("taifex_vix_min_new", "https://www.taifex.com.tw/cht/7/vixMinNew"),
]

KEYWORDS = [
    "vix",
    "volatility",
    "put/call",
    "pc ratio",
    "三大法人",
    "institution",
    "本益比",
    "殖利率",
    "股價淨值比",
    "margin",
    "financing",
]


def probe_risk_sources(registry_path: Path, out_dir: Path, insecure: bool = False) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = _probe_targets(registry_path)
    rows = [probe_url(source_name, url, insecure=insecure) for source_name, url in targets]
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "risk_source_probe.csv", index=False)
    (out_dir / "risk_source_probe.md").write_text(_probe_markdown(out), encoding="utf-8")
    return out


def probe_url(source_name: str, url: str, insecure: bool = False) -> dict[str, Any]:
    base = {
        "source_name": source_name,
        "url": url,
        "status_code": "",
        "content_type": "",
        "encoding": "",
        "response_size": "",
        "first_500_chars": "",
        "detected_format": "UNKNOWN",
        "json_parse_success": False,
        "csv_parse_success": False,
        "html_table_count": 0,
        "sample_columns": "",
        "sample_row_count": "",
        "sample_date_min": "",
        "sample_date_max": "",
        "query_params_supported_guess": "",
        "swagger_related_paths": "",
        "error_message": "",
    }
    try:
        response = _fetch(url, insecure=insecure)
    except ssl.SSLError as exc:
        return {**base, "detected_format": "SSL_ERROR", "error_message": f"SSL_ERROR: {exc}"}
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        if isinstance(reason, ssl.SSLError) or "CERTIFICATE_VERIFY_FAILED" in str(reason):
            return {**base, "detected_format": "SSL_ERROR", "error_message": f"SSL_ERROR: {reason}"}
        return {**base, "error_message": f"URL_ERROR: {reason}"}
    except Exception as exc:  # noqa: BLE001 - one probe failure must not stop all probes.
        return {**base, "error_message": f"{type(exc).__name__}: {exc}"}

    text = response["text"]
    detected = detect_format(text, response["content_type"])
    parsed = _parse_sample(text, detected)
    swagger_paths = ""
    if detected == "JSON" and "swagger" in source_name:
        swagger_paths = ";".join(extract_swagger_related_paths(_json_or_none(text)))
    return {
        **base,
        "status_code": response["status_code"],
        "content_type": response["content_type"],
        "encoding": response["encoding"],
        "response_size": len(response["body"]),
        "first_500_chars": text[:500].replace("\n", " ").replace("\r", " "),
        "detected_format": detected,
        **parsed,
        "query_params_supported_guess": _query_guess(url, text),
        "swagger_related_paths": swagger_paths,
    }


def detect_format(text: str, content_type: str = "") -> str:
    ctype = content_type.lower()
    stripped = text.lstrip()
    if "json" in ctype or stripped.startswith("{") or stripped.startswith("["):
        return "JSON"
    if "csv" in ctype:
        return "CSV"
    if "<table" in stripped.lower() and ("<html" in stripped.lower() or "<!doctype" in stripped.lower()):
        return "EXCEL_HTML"
    if "<html" in stripped.lower() or "<!doctype" in stripped.lower():
        return "HTML"
    if "," in text[:1000] and "\n" in text[:1000]:
        return "CSV"
    return "UNKNOWN"


def extract_swagger_related_paths(spec: Any) -> list[str]:
    if not isinstance(spec, dict):
        return []
    paths = spec.get("paths", {})
    if not isinstance(paths, dict):
        return []
    out = []
    for path, meta in paths.items():
        blob = f"{path} {json.dumps(meta, ensure_ascii=False)}".lower()
        if any(keyword.lower() in blob for keyword in KEYWORDS):
            out.append(str(path))
    return out


def _probe_targets(registry_path: Path) -> list[tuple[str, str]]:
    targets: list[tuple[str, str]] = []
    for source in read_source_registry(registry_path):
        url = source.api_endpoint or source.url
        if url:
            targets.append((source.source_name, url))
    targets.extend(SWAGGER_URLS)
    targets.extend(KNOWN_ENDPOINTS)
    seen = set()
    unique = []
    for item in targets:
        if item not in seen:
            unique.append(item)
            seen.add(item)
    return unique


def _fetch(url: str, insecure: bool = False) -> dict[str, Any]:
    context = ssl._create_unverified_context() if insecure else None
    request = urllib.request.Request(url, headers={"User-Agent": "txo-risk-source-probe/1.0"})
    with urllib.request.urlopen(request, timeout=30, context=context) as response:
        body = response.read()
        encoding = response.headers.get_content_charset() or "utf-8"
        try:
            text = body.decode(encoding, errors="replace")
        except LookupError:
            encoding = "utf-8"
            text = body.decode(encoding, errors="replace")
        return {
            "status_code": getattr(response, "status", ""),
            "content_type": response.headers.get("Content-Type", ""),
            "encoding": encoding,
            "body": body,
            "text": text,
        }


def _parse_sample(text: str, detected: str) -> dict[str, Any]:
    if detected == "JSON":
        data = _json_or_none(text)
        if data is None:
            return {"json_parse_success": False}
        frame = pd.DataFrame(data if isinstance(data, list) else data.get("data", []))
        return {"json_parse_success": True, **_sample_frame(frame)}
    if detected == "CSV":
        try:
            frame = pd.read_csv(io.StringIO(text))
            return {"csv_parse_success": True, **_sample_frame(frame)}
        except Exception as exc:  # noqa: BLE001
            return {"csv_parse_success": False, "error_message": f"CSV_PARSE_ERROR: {exc}"}
    if detected in {"HTML", "EXCEL_HTML"}:
        try:
            tables = pd.read_html(io.StringIO(text))
            frame = tables[0] if tables else pd.DataFrame()
            return {"html_table_count": len(tables), **_sample_frame(frame)}
        except Exception:
            return {"html_table_count": 0}
    return {}


def _sample_frame(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"sample_columns": "", "sample_row_count": 0}
    date_values = _find_date_series(frame)
    return {
        "sample_columns": ";".join(map(str, frame.columns[:25])),
        "sample_row_count": int(len(frame)),
        "sample_date_min": str(date_values.min().date()) if date_values is not None and date_values.notna().any() else "",
        "sample_date_max": str(date_values.max().date()) if date_values is not None and date_values.notna().any() else "",
    }


def _find_date_series(frame: pd.DataFrame) -> pd.Series | None:
    for col in frame.columns:
        name = str(col).lower()
        if "date" in name or "日期" in name:
            parsed = pd.to_datetime(frame[col].astype(str), errors="coerce")
            compact = parsed.isna() & frame[col].astype(str).str.match(r"^\d{8}$", na=False)
            if compact.any():
                parsed.loc[compact] = pd.to_datetime(frame.loc[compact, col].astype(str), format="%Y%m%d", errors="coerce")
            return parsed
    return None


def _json_or_none(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return None


def _query_guess(url: str, text: str) -> str:
    lowered = text.lower()
    if "startdate" in lowered or "enddate" in lowered or "query" in lowered or "parameters" in lowered:
        return "possible"
    if "?" in url:
        return "url_has_query"
    return "unknown"


def _probe_markdown(probe: pd.DataFrame) -> str:
    lines = [
        "# Risk Source Probe",
        "",
        "This report probes official or registered endpoints only. It does not write processed risk indicators or change strategies.",
        "",
        "## Endpoint Results",
        "",
    ]
    if probe.empty:
        lines.append("- No endpoints probed.")
    else:
        for row in probe.itertuples(index=False):
            lines.append(f"- {row.source_name}: format={row.detected_format}, status={row.status_code}, rows={row.sample_row_count}, error={row.error_message}")
    swagger = probe[probe["swagger_related_paths"].astype(str) != ""] if not probe.empty and "swagger_related_paths" in probe else pd.DataFrame()
    lines.extend(["", "## Swagger Related Paths", ""])
    if swagger.empty:
        lines.append("- None found.")
    else:
        for row in swagger.itertuples(index=False):
            lines.append(f"- {row.source_name}: {row.swagger_related_paths}")
    lines.extend(
        [
            "",
            "## Required Limitations",
            "",
            "- SSL failures are recorded and not bypassed unless --insecure is explicitly used.",
            "- HTML is not treated as JSON.",
            "- No paid or unknown sources are added.",
        ]
    )
    return "\n".join(lines) + "\n"
