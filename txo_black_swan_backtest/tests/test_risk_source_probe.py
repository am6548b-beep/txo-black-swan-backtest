from __future__ import annotations

import ssl
import urllib.error

import pandas as pd

from src.risk_source_probe import detect_format, extract_swagger_related_paths, probe_url, _probe_markdown


def test_probe_does_not_crash_on_single_source_failure(monkeypatch) -> None:
    def boom(url: str, insecure: bool = False):
        raise urllib.error.URLError("network down")

    monkeypatch.setattr("src.risk_source_probe._fetch", boom)
    row = probe_url("bad_source", "https://example.invalid")

    assert row["error_message"].startswith("URL_ERROR")


def test_html_response_not_treated_as_json() -> None:
    html = "<html><body><table><tr><th>Date</th></tr><tr><td>2024-01-01</td></tr></table></body></html>"

    assert detect_format(html, "text/html") == "EXCEL_HTML"


def test_ssl_error_is_recorded(monkeypatch) -> None:
    def ssl_fail(url: str, insecure: bool = False):
        raise urllib.error.URLError(ssl.SSLError("certificate verify failed"))

    monkeypatch.setattr("src.risk_source_probe._fetch", ssl_fail)
    row = probe_url("ssl_source", "https://example.com")

    assert row["detected_format"] == "SSL_ERROR"
    assert "SSL_ERROR" in row["error_message"]


def test_probe_report_has_no_best_or_recommend() -> None:
    text = _probe_markdown(pd.DataFrame()).lower()

    assert "best" not in text
    assert "recommend" not in text


def test_swagger_path_extraction_handles_empty_result() -> None:
    assert extract_swagger_related_paths({"paths": {"/foo": {"get": {"summary": "unrelated"}}}}) == []
    assert extract_swagger_related_paths({}) == []
