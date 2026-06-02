"""Prepare raw TAIFEX/CrazyIndicator data into processed CSV files.

This script only performs data cleaning, normalization, and audit reporting.
It does not run a backtest, estimate IV/delta, optimize parameters, or modify
strategy/execution logic.
"""

from __future__ import annotations

import argparse
import calendar
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ENCODINGS = ("utf-8-sig", "utf-8", "cp950", "big5")

MARKET_COLUMNS = [
    "date",
    "tx_open",
    "tx_high",
    "tx_low",
    "tx_close",
    "txf_close",
    "volume",
    "vix",
    "event_flag",
    "data_source",
    "is_official_taifex",
    "is_backfilled",
    "source_file",
]

OPTION_COLUMNS = [
    "date",
    "contract",
    "expiry",
    "dte",
    "cp",
    "strike",
    "open",
    "high",
    "low",
    "close",
    "bid",
    "ask",
    "volume",
    "open_interest",
    "settlement_price",
    "halt_flag",
    "session",
    "is_weekly",
    "iv",
    "delta",
    "iv_estimated",
    "delta_estimated",
    "bid_ask_estimated",
    "quote_quality_status",
    "spread_pct",
    "is_tradable_quote",
    "data_source",
    "source_file",
]

DATE_ALIASES = ["\u4ea4\u6613\u65e5\u671f", "\u65e5\u671f", "Date", "TradingDate"]
CONTRACT_ALIASES = ["\u5951\u7d04", "\u5546\u54c1", "\u5951\u7d04\u540d\u7a31", "\u5546\u54c1\u4ee3\u865f", "Contract"]
EXPIRY_ALIASES = ["\u5230\u671f\u6708\u4efd(\u9031\u5225)", "\u5230\u671f\u6708\u4efd", "\u5230\u671f\u6708\u4efd(\u9031\u5225/\u6708\u4efd)", "\u5230\u671f\u5e74\u6708", "Expiry", "ContractMonth"]
STRIKE_ALIASES = ["\u5c65\u7d04\u50f9", "\u5c65\u7d04\u50f9\u683c", "StrikePrice", "Strike"]
CP_ALIASES = ["\u8cb7\u8ce3\u6b0a", "\u8cb7\u8ce3\u6b0a\u5225", "CallPut", "CP"]
OPEN_ALIASES = ["\u958b\u76e4\u50f9", "\u958b\u76e4", "Open"]
HIGH_ALIASES = ["\u6700\u9ad8\u50f9", "\u6700\u9ad8", "High"]
LOW_ALIASES = ["\u6700\u4f4e\u50f9", "\u6700\u4f4e", "Low"]
CLOSE_ALIASES = ["\u6700\u5f8c\u6210\u4ea4\u50f9", "\u6536\u76e4\u50f9", "\u6536\u76e4", "Close", "LastPrice"]
VOLUME_ALIASES = ["\u6210\u4ea4\u91cf", "Volume"]
SETTLEMENT_ALIASES = ["\u7d50\u7b97\u50f9", "SettlementPrice", "Settlement"]
OPEN_INTEREST_ALIASES = ["\u672a\u6c96\u92b7\u5951\u7d04\u91cf", "\u672a\u6c96\u92b7\u5951\u7d04\u6578", "\u672a\u5e73\u5009\u91cf", "OpenInterest", "OI"]
BID_ALIASES = ["\u6700\u5f8c\u6700\u4f73\u8cb7\u50f9", "\u6700\u4f73\u8cb7\u50f9", "\u8cb7\u50f9", "Bid"]
ASK_ALIASES = ["\u6700\u5f8c\u6700\u4f73\u8ce3\u50f9", "\u6700\u4f73\u8ce3\u50f9", "\u8ce3\u50f9", "Ask"]
HALT_ALIASES = ["\u662f\u5426\u56e0\u8a0a\u606f\u9762\u66ab\u505c\u4ea4\u6613", "\u66ab\u505c\u4ea4\u6613", "HaltFlag"]
SESSION_ALIASES = ["\u4ea4\u6613\u6642\u6bb5", "\u6642\u6bb5", "Session"]
TXO_CONTRACT_ALIASES = {"TXO", "CAO"}
SCHEMA_AUDIT_ROWS: list[dict[str, object]] = []


@dataclass(frozen=True)
class AuditItem:
    dataset: str
    check: str
    status: str
    detail: str


@dataclass(frozen=True)
class ExpiryCalendar:
    path: Path
    exists: bool
    overrides: dict[str, pd.Timestamp]
    warnings: tuple[str, ...]


def main() -> None:
    global SCHEMA_AUDIT_ROWS
    SCHEMA_AUDIT_ROWS = []
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    report_dir = Path(args.report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    expiry_calendar = load_expiry_calendar(raw_dir / "taifex" / "txo_expiry_calendar.csv")
    txf_market = load_txf_1min(raw_dir / "taifex" / "txf")
    official_market = load_official_futures(raw_dir / "taifex" / "fut")
    market = merge_market_sources(txf_market, official_market)
    options = load_official_options(raw_dir / "taifex" / "opt", expiry_calendar)

    market.to_csv(out_dir / "market.csv", index=False)
    options[[col for col in OPTION_COLUMNS if col in options.columns]].to_csv(out_dir / "options.csv", index=False)

    audit = build_audit(market, options, txf_market, official_market, expiry_calendar)
    audit_df = pd.DataFrame([item.__dict__ for item in audit])
    audit_df.to_csv(report_dir / "data_cleaning_audit.csv", index=False)
    schema_audit = pd.DataFrame(SCHEMA_AUDIT_ROWS)
    schema_audit.to_csv(report_dir / "schema_mapping_audit.csv", index=False)
    write_schema_mapping_audit_markdown(report_dir / "schema_mapping_audit.md", schema_audit)
    debug_counts = write_debug_outputs(report_dir, options)
    write_quote_quality_reports(report_dir, options)
    raw_inventory = build_raw_options_file_inventory(raw_dir / "taifex" / "opt", options)
    raw_inventory.to_csv(report_dir / "raw_options_file_inventory.csv", index=False)
    write_raw_options_inventory_markdown(report_dir / "raw_options_file_inventory.md", raw_inventory)
    recognition_debug = build_raw_options_file_recognition_debug(raw_dir / "taifex" / "opt", options)
    recognition_debug.to_csv(report_dir / "raw_options_file_recognition_debug.csv", index=False)
    write_raw_options_recognition_debug_markdown(report_dir / "raw_options_file_recognition_debug.md", recognition_debug)
    write_audit_markdown(report_dir / "data_cleaning_audit.md", audit_df, market, options)

    counts = audit_df["status"].value_counts().to_dict() if not audit_df.empty else {}
    print(f"market rows: {len(market)}")
    print(f"options rows: {len(options)}")
    print(f"data_cleaning_audit status counts: {counts}")
    print(f"wrote {out_dir / 'market.csv'}")
    print(f"wrote {out_dir / 'options.csv'}")
    print(f"wrote {report_dir / 'data_cleaning_audit.csv'}")
    print(f"wrote {report_dir / 'data_cleaning_audit.md'}")
    print(f"wrote {report_dir / 'schema_mapping_audit.csv'}")
    print(f"wrote {report_dir / 'schema_mapping_audit.md'}")
    print(f"wrote {report_dir / 'raw_options_file_inventory.csv'}")
    print(f"wrote {report_dir / 'raw_options_file_inventory.md'}")
    print(f"wrote {report_dir / 'raw_options_file_recognition_debug.csv'}")
    print(f"wrote {report_dir / 'raw_options_file_recognition_debug.md'}")
    print(f"bad expiry rows: {debug_counts['bad_expiry_rows']}")
    print(f"bad bid/ask rows: {debug_counts['bad_bid_ask_rows']}")
    print(f"extreme spread sample rows: {debug_counts['extreme_spread_rows_sample']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare raw TAIFEX data into processed CSV files")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--report-dir", default="reports")
    return parser.parse_args()


def load_txf_1min(txf_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(txf_dir.glob("*.csv")):
        df = read_csv_with_encoding(path)
        if df.empty:
            continue
        df = normalize_columns(df)
        required = {"Date", "Time", "Open", "High", "Low", "Close", "Volume"}
        if not required <= set(df.columns):
            continue
        work = pd.DataFrame()
        work["date"] = parse_date(df["Date"])
        work["time"] = df["Time"].astype(str)
        for src, dst in [("Open", "open"), ("High", "high"), ("Low", "low"), ("Close", "close"), ("Volume", "volume")]:
            work[dst] = to_number(df[src])
        work["source_file"] = path.name
        work = work.dropna(subset=["date"]).sort_values(["date", "time"]).reset_index(drop=True)
        if work.empty:
            continue
        daily = (
            work.groupby("date", as_index=False)
            .agg(
                tx_open=("open", "first"),
                tx_high=("high", "max"),
                tx_low=("low", "min"),
                tx_close=("close", "last"),
                txf_close=("close", "last"),
                volume=("volume", "sum"),
                source_file=("source_file", "first"),
            )
        )
        daily["vix"] = np.nan
        daily["event_flag"] = 0
        daily["data_source"] = "crazyindicator_1min"
        daily["is_official_taifex"] = False
        daily["is_backfilled"] = False
        frames.append(daily[MARKET_COLUMNS])
    return concat_or_empty(frames, MARKET_COLUMNS).sort_values("date").reset_index(drop=True)


def load_official_futures(fut_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(fut_dir.glob("*.csv")):
        df = read_csv_with_encoding(path)
        if df.empty:
            continue
        df = normalize_columns(df)
        date_col = first_col(df, DATE_ALIASES)
        contract_col = first_col(df, CONTRACT_ALIASES)
        expiry_col = first_col(df, EXPIRY_ALIASES)
        if date_col is None or contract_col is None:
            continue
        tx = df[df[contract_col].astype(str).str.strip().str.upper().eq("TX")].copy()
        if tx.empty:
            continue
        work = pd.DataFrame()
        work["date"] = parse_date(tx[date_col])
        work["contract"] = tx[contract_col].astype(str).str.strip()
        work["expiry_raw"] = tx[expiry_col].astype(str).str.strip() if expiry_col else ""
        work["expiry_key"] = expiry_sort_key(work["expiry_raw"])
        work["tx_open"] = to_number(col_or_nan(tx, OPEN_ALIASES))
        work["tx_high"] = to_number(col_or_nan(tx, HIGH_ALIASES))
        work["tx_low"] = to_number(col_or_nan(tx, LOW_ALIASES))
        work["tx_close"] = to_number(col_or_nan(tx, CLOSE_ALIASES))
        work["txf_close"] = work["tx_close"]
        work["volume"] = to_number(col_or_nan(tx, VOLUME_ALIASES)).fillna(0)
        work["source_file"] = path.name
        work = work.dropna(subset=["date"]).sort_values(["date", "expiry_key", "volume"], ascending=[True, True, False])
        selected = work.groupby("date", as_index=False).first()
        selected["vix"] = np.nan
        selected["event_flag"] = 0
        selected["data_source"] = "taifex_official_daily"
        selected["is_official_taifex"] = True
        selected["is_backfilled"] = True
        frames.append(selected[MARKET_COLUMNS])
    return concat_or_empty(frames, MARKET_COLUMNS).sort_values("date").reset_index(drop=True)


def merge_market_sources(txf_market: pd.DataFrame, official_market: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([txf_market, official_market], ignore_index=True)
    if combined.empty:
        return pd.DataFrame(columns=MARKET_COLUMNS)
    combined["source_priority"] = combined["is_official_taifex"].map({True: 1, False: 0}).fillna(0)
    combined = combined.sort_values(["date", "source_priority"]).drop_duplicates("date", keep="last")
    return combined.drop(columns=["source_priority"]).sort_values("date").reset_index(drop=True)[MARKET_COLUMNS]


def load_official_options(opt_dir: Path, expiry_calendar: ExpiryCalendar | None = None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(opt_dir.glob("*.csv")):
        df = read_csv_with_encoding(path)
        if df.empty:
            append_schema_audit(path, df, "unknown", {}, ["date", "contract", "expiry_raw", "strike", "cp"], 0, "FAIL", "empty_or_unreadable")
            continue
        df = normalize_columns(df)
        schema = detect_option_schema(df)
        date_col = schema.get("date") or first_col(df, ["鈭斗??交?", "?交?"])
        if date_col is None:
            append_schema_audit(path, df, "unknown", schema, ["date"], 0, "FAIL", "missing date column")
            continue
        contract_col = schema.get("contract") or first_col(df, ["憟?"])
        if contract_col is None:
            append_schema_audit(path, df, "unknown", schema, ["contract"], 0, "FAIL", "missing contract column")
            continue
        contract = df[contract_col].astype(str).str.strip().str.upper()
        txo = df[contract.isin(TXO_CONTRACT_ALIASES)].copy()
        if txo.empty:
            append_schema_audit(path, df, schema_version(schema), schema, [], 0, "FAIL", "no TXO/CAO rows")
            continue
        required_missing = [name for name in ["date", "contract", "expiry_raw", "strike", "cp"] if schema.get(name) is None]
        if required_missing:
            append_schema_audit(path, df, schema_version(schema), schema, required_missing, 0, "FAIL", "missing required columns")
            continue
        out = pd.DataFrame()
        out["date"] = parse_date(txo[date_col])
        out["contract"] = txo[contract_col].astype(str).str.strip()
        expiry_raw = txo[schema["expiry_raw"]].astype(str).str.strip()
        out["expiry_raw"] = expiry_raw
        rule_lookup = {raw: parse_expiry_by_rule(raw) for raw in expiry_raw.dropna().unique()}
        expiry_lookup = {raw: parse_taifex_expiry(raw, expiry_calendar.overrides if expiry_calendar else None) for raw in expiry_raw.dropna().unique()}
        out["rule_expiry"] = expiry_raw.map(rule_lookup)
        out["expiry"] = expiry_raw.map(expiry_lookup)
        out["dte"] = (pd.to_datetime(out["expiry"]) - pd.to_datetime(out["date"])).dt.days
        out["rule_dte"] = (pd.to_datetime(out["rule_expiry"]) - pd.to_datetime(out["date"])).dt.days
        out["override_used"] = [
            bool(raw in expiry_calendar.overrides and pd.notna(expiry_calendar.overrides[raw])) if expiry_calendar else False
            for raw in expiry_raw
        ]
        out["cp"] = txo[schema["cp"]].map(normalize_cp)
        out["strike"] = to_number(txo[schema["strike"]])
        out["open"] = to_number(txo[schema["open"]]) if schema.get("open") else np.nan
        out["high"] = to_number(txo[schema["high"]]) if schema.get("high") else np.nan
        out["low"] = to_number(txo[schema["low"]]) if schema.get("low") else np.nan
        out["close"] = to_number(txo[schema["close"]]) if schema.get("close") else np.nan
        out["bid"] = to_number(txo[schema["bid"]]) if schema.get("bid") else np.nan
        out["ask"] = to_number(txo[schema["ask"]]) if schema.get("ask") else np.nan
        out["volume"] = (to_number(txo[schema["volume"]]) if schema.get("volume") else pd.Series(np.nan, index=txo.index)).fillna(0)
        out["open_interest"] = to_number(txo[schema["open_interest"]]) if schema.get("open_interest") else np.nan
        out["settlement_price"] = to_number(txo[schema["settlement_price"]]) if schema.get("settlement_price") else np.nan
        out["halt_flag"] = txo[schema["halt_flag"]] if schema.get("halt_flag") else ""
        out["session"] = txo[schema["session"]] if schema.get("session") else ""
        out["is_weekly"] = expiry_raw.str.contains("W", case=False, na=False)
        out["iv"] = np.nan
        out["delta"] = np.nan
        out["iv_estimated"] = False
        out["delta_estimated"] = False
        out["bid_ask_estimated"] = False
        out = add_quote_quality_columns(out)
        out["data_source"] = "taifex_official_daily"
        out["source_file"] = path.name
        append_schema_audit(path, df, schema_version(schema), schema, [], len(out), "PASS", "")
        frames.append(out[OPTION_COLUMNS + ["expiry_raw", "rule_expiry", "rule_dte", "override_used"]])
    return concat_or_empty(frames, OPTION_COLUMNS + ["expiry_raw", "rule_expiry", "rule_dte", "override_used"]).reset_index(drop=True)


def detect_option_schema(df: pd.DataFrame) -> dict[str, str | None]:
    return {
        "date": first_col(df, DATE_ALIASES),
        "contract": first_col(df, CONTRACT_ALIASES),
        "expiry_raw": first_col(df, EXPIRY_ALIASES),
        "strike": first_col(df, STRIKE_ALIASES),
        "cp": first_col(df, CP_ALIASES),
        "open": first_col(df, OPEN_ALIASES),
        "high": first_col(df, HIGH_ALIASES),
        "low": first_col(df, LOW_ALIASES),
        "close": first_col(df, CLOSE_ALIASES),
        "volume": first_col(df, VOLUME_ALIASES),
        "settlement_price": first_col(df, SETTLEMENT_ALIASES),
        "open_interest": first_col(df, OPEN_INTEREST_ALIASES),
        "bid": first_col(df, BID_ALIASES),
        "ask": first_col(df, ASK_ALIASES),
        "halt_flag": first_col(df, HALT_ALIASES),
        "session": first_col(df, SESSION_ALIASES),
    }


def schema_version(schema: dict[str, str | None]) -> str:
    mapped = {value for value in schema.values() if value}
    if any(value in mapped for value in DATE_ALIASES + CONTRACT_ALIASES):
        return "alias_schema"
    return "legacy_schema"


def append_schema_audit(
    path: Path,
    df: pd.DataFrame,
    detected_schema_version: str,
    schema: dict[str, str | None],
    missing_required_columns: list[str],
    normalized_rows: int,
    status: str,
    issue: str,
) -> None:
    dates = parse_date(df[schema["date"]]) if schema.get("date") and schema["date"] in df else pd.Series(dtype="datetime64[ns]")
    dates = dates.dropna()
    years = sorted({int(year) for year in dates.dt.year.dropna().unique()}) if not dates.empty else []
    SCHEMA_AUDIT_ROWS.append(
        {
            "source_file": path.name,
            "detected_schema_version": detected_schema_version,
            "mapped_columns": ";".join(f"{key}={value}" for key, value in schema.items() if value),
            "missing_required_columns": ";".join(missing_required_columns),
            "normalized_rows": int(normalized_rows),
            "date_min": str(dates.min().date()) if not dates.empty else "",
            "date_max": str(dates.max().date()) if not dates.empty else "",
            "years_covered": ";".join(map(str, years)),
            "status": status,
            "issue": issue,
        }
    )


def write_schema_mapping_audit_markdown(path: Path, audit: pd.DataFrame) -> None:
    counts = audit["status"].value_counts().to_dict() if not audit.empty else {}
    lines = [
        "# Schema Mapping Audit",
        "",
        "This report audits raw option schema mapping only. It does not modify strategy rules, prices, quote quality, formulas, or trades.",
        "",
        "## Status Counts",
        "",
    ]
    for status in ["PASS", "WARN", "FAIL"]:
        lines.append(f"- {status}: {int(counts.get(status, 0))}")
    lines.extend(["", "## Failures", ""])
    failures = audit[audit["status"] == "FAIL"] if not audit.empty else pd.DataFrame()
    if failures.empty:
        lines.append("- none")
    else:
        for row in failures.head(50).itertuples(index=False):
            lines.append(f"- {row.source_file}: missing={row.missing_required_columns}, issue={row.issue}")
    lines.extend(
        [
            "",
            "## Required Limitations",
            "",
            "- Existing legacy schema support is retained.",
            "- No raw rows are deleted.",
            "- No bid/ask repair or price filling is performed.",
            "- No backtest is run by this audit.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_audit(
    market: pd.DataFrame,
    options: pd.DataFrame,
    txf_market: pd.DataFrame,
    official_market: pd.DataFrame,
    expiry_calendar: ExpiryCalendar,
) -> list[AuditItem]:
    rows: list[AuditItem] = []
    rows.extend(audit_market("market", market))
    rows.extend(audit_options("options", options))
    rows.append(AuditItem("market", "source_file_distribution", "PASS", distribution(market, "source_file")))
    rows.append(AuditItem("market", "data_source_distribution", "PASS", distribution(market, "data_source")))
    rows.append(AuditItem("market", "official_rows", "PASS" if len(official_market) > 0 else "WARN", f"rows={len(official_market)}"))
    rows.append(AuditItem("market", "crazyindicator_rows", "PASS" if len(txf_market) > 0 else "WARN", f"rows={len(txf_market)}"))
    rows.append(AuditItem("options", "weekly_monthly_distribution", "PASS", distribution(options, "is_weekly")))
    rows.append(AuditItem("options", "source_file_distribution", "PASS", distribution(options, "source_file")))
    rows.extend(audit_expiry_calendar(options, expiry_calendar))
    return rows


def audit_market(name: str, df: pd.DataFrame) -> list[AuditItem]:
    rows = basic_audit(name, df, ["date", "tx_open", "tx_high", "tx_low", "tx_close", "txf_close", "volume"])
    if df.empty:
        return rows
    rows.extend(ohlc_audit(name, df, "tx_open", "tx_high", "tx_low", "tx_close"))
    rows.append(nonnegative_check(name, "volume_nonnegative", df["volume"]))
    rows.extend(date_gap_audit(name, df["date"]))
    return rows


def audit_options(name: str, df: pd.DataFrame) -> list[AuditItem]:
    required = ["date", "contract", "expiry", "dte", "cp", "strike", "close", "bid", "ask", "volume", "open_interest"]
    rows = basic_audit(name, df, required)
    if df.empty:
        return rows
    rows.extend(ohlc_audit(name, df, "open", "high", "low", "close"))
    rows.append(nonnegative_check(name, "volume_nonnegative", df["volume"]))
    rows.append(nonnegative_check(name, "open_interest_nonnegative", df["open_interest"]))
    rows.append(AuditItem(name, "bid_le_ask", "FAIL" if ((df["bid"].notna()) & (df["ask"].notna()) & (df["bid"] > df["ask"])).any() else "PASS", f"bad_rows={int(((df['bid'].notna()) & (df['ask'].notna()) & (df['bid'] > df['ask'])).sum())}"))
    bad_bidask_negative = ((df["bid"].notna()) & (df["bid"] < 0)) | ((df["ask"].notna()) & (df["ask"] < 0))
    rows.append(AuditItem(name, "bid_ask_nonnegative", "FAIL" if bad_bidask_negative.any() else "PASS", f"bad_rows={int(bad_bidask_negative.sum())}"))
    expiry_bad = (pd.to_datetime(df["expiry"], errors="coerce").notna()) & (pd.to_datetime(df["date"], errors="coerce").notna()) & (pd.to_datetime(df["expiry"]) < pd.to_datetime(df["date"]))
    rows.append(AuditItem(name, "expiry_ge_date", "FAIL" if expiry_bad.any() else "PASS", f"bad_rows={int(expiry_bad.sum())}"))
    rows.append(AuditItem(name, "dte_nonnegative", "FAIL" if (df["dte"].notna() & (df["dte"] < 0)).any() else "PASS", f"bad_rows={int((df['dte'].notna() & (df['dte'] < 0)).sum())}"))
    spread = spread_pct(df)
    gt_05 = int((spread > 0.5).sum())
    gt_10 = int((spread > 1.0).sum())
    rows.append(AuditItem(name, "spread_pct_gt_0_5", "WARN" if gt_05 else "PASS", f"rows={gt_05}"))
    rows.append(AuditItem(name, "spread_pct_gt_1_0", "FAIL" if gt_10 else "PASS", f"rows={gt_10}"))
    unparsed = int(df["expiry"].isna().sum())
    rows.append(AuditItem(name, "unparsed_expiry", "WARN" if unparsed else "PASS", f"rows={unparsed}"))
    invalid_cp = int(~df["cp"].isin(["C", "P"]).sum()) if False else int((~df["cp"].isin(["C", "P"])).sum())
    rows.append(AuditItem(name, "invalid_cp", "WARN" if invalid_cp else "PASS", f"rows={invalid_cp}"))
    rows.extend(date_gap_audit(name, df["date"].drop_duplicates()))
    contract_keys = ["date", "contract", "expiry", "cp", "strike", "session"]
    dup_contracts = int(df.duplicated(contract_keys, keep=False).sum())
    rows.append(AuditItem(name, "duplicate_contracts", "WARN" if dup_contracts else "PASS", f"rows={dup_contracts}"))
    return rows


def audit_expiry_calendar(options: pd.DataFrame, expiry_calendar: ExpiryCalendar) -> list[AuditItem]:
    override_used = int(options["override_used"].fillna(False).sum()) if "override_used" in options else 0
    before = bad_expiry_count(options, "rule_expiry", "rule_dte")
    after = bad_expiry_count(options, "expiry", "dte")
    warnings = len(expiry_calendar.warnings)
    return [
        AuditItem(
            "expiry_calendar",
            "override_file_exists",
            "PASS" if expiry_calendar.exists else "WARN",
            str(expiry_calendar.path) if expiry_calendar.exists else f"missing: {expiry_calendar.path}",
        ),
        AuditItem("expiry_calendar", "override_used_count", "PASS", f"rows={override_used}"),
        AuditItem("expiry_calendar", "bad_expiry_before_override", "WARN" if before else "PASS", f"rows={before}"),
        AuditItem("expiry_calendar", "bad_expiry_after_override", "WARN" if after else "PASS", f"rows={after}"),
        AuditItem(
            "expiry_calendar",
            "override_warnings",
            "WARN" if warnings else "PASS",
            "; ".join(expiry_calendar.warnings[:20]) if warnings else "none",
        ),
    ]


def add_quote_quality_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    bid = pd.to_numeric(out["bid"], errors="coerce")
    ask = pd.to_numeric(out["ask"], errors="coerce")
    mid = (bid + ask) / 2.0
    spread = (ask - bid) / mid.replace(0, np.nan)
    status = pd.Series("VALID", index=out.index, dtype="object")
    status[spread > 0.5] = "EXTREME_SPREAD_WARN"
    status[spread > 1.0] = "EXTREME_SPREAD_FAIL"
    status[bid > ask] = "BID_GT_ASK"
    status[ask == 0] = "ZERO_ASK"
    status[bid == 0] = "ZERO_BID"
    status[(bid < 0) | (ask < 0)] = "NEGATIVE_QUOTE"
    status[ask.isna()] = "MISSING_ASK"
    status[bid.isna()] = "MISSING_BID"
    out["quote_quality_status"] = status
    out["spread_pct"] = spread.where(status.isin(["VALID", "EXTREME_SPREAD_WARN", "EXTREME_SPREAD_FAIL"]))
    out["is_tradable_quote"] = out["quote_quality_status"].eq("VALID")
    return out


def classify_quote_quality(bid: object, ask: object) -> tuple[str, float]:
    bid_value = pd.to_numeric(pd.Series([bid]), errors="coerce").iloc[0]
    ask_value = pd.to_numeric(pd.Series([ask]), errors="coerce").iloc[0]
    if pd.isna(bid_value):
        return "MISSING_BID", np.nan
    if pd.isna(ask_value):
        return "MISSING_ASK", np.nan
    bid_float = float(bid_value)
    ask_float = float(ask_value)
    if bid_float < 0 or ask_float < 0:
        return "NEGATIVE_QUOTE", np.nan
    if bid_float == 0:
        return "ZERO_BID", np.nan
    if ask_float == 0:
        return "ZERO_ASK", np.nan
    if bid_float > ask_float:
        return "BID_GT_ASK", np.nan
    mid = (bid_float + ask_float) / 2.0
    spread = np.nan if mid == 0 else (ask_float - bid_float) / mid
    if pd.notna(spread) and spread > 1.0:
        return "EXTREME_SPREAD_FAIL", float(spread)
    if pd.notna(spread) and spread > 0.5:
        return "EXTREME_SPREAD_WARN", float(spread)
    return "VALID", float(spread)


def bad_expiry_count(options: pd.DataFrame, expiry_col: str, dte_col: str) -> int:
    if options.empty or expiry_col not in options or dte_col not in options:
        return 0
    date = pd.to_datetime(options["date"], errors="coerce")
    expiry = pd.to_datetime(options[expiry_col], errors="coerce")
    mask = ((expiry.notna()) & (date.notna()) & (expiry < date)) | (options[dte_col].notna() & (options[dte_col] < 0))
    return int(mask.sum())


def basic_audit(name: str, df: pd.DataFrame, required: list[str]) -> list[AuditItem]:
    rows = [
        AuditItem(name, "row_count", "PASS" if len(df) else "WARN", f"rows={len(df)}"),
        AuditItem(name, "date_range", "PASS" if "date" in df and df["date"].notna().any() else "WARN", date_range_detail(df)),
        AuditItem(name, "duplicate_rows", "WARN" if int(df.duplicated().sum()) else "PASS", f"rows={int(df.duplicated().sum())}"),
    ]
    missing = [col for col in required if col not in df.columns]
    rows.append(AuditItem(name, "missing_required_columns", "FAIL" if missing else "PASS", ",".join(missing)))
    return rows


def ohlc_audit(name: str, df: pd.DataFrame, open_col: str, high_col: str, low_col: str, close_col: str) -> list[AuditItem]:
    cols = [open_col, high_col, low_col, close_col]
    missing = [col for col in cols if col not in df.columns]
    if missing:
        return [AuditItem(name, "ohlc_numeric", "FAIL", f"missing={','.join(missing)}")]
    numeric_bad = int(sum(df[col].notna().sum() - pd.to_numeric(df[col], errors="coerce").notna().sum() for col in cols))
    high_bad = int((df[high_col].notna() & ((df[high_col] < df[open_col]) | (df[high_col] < df[close_col]) | (df[high_col] < df[low_col]))).sum())
    low_bad = int((df[low_col].notna() & ((df[low_col] > df[open_col]) | (df[low_col] > df[close_col]) | (df[low_col] > df[high_col]))).sum())
    return [
        AuditItem(name, "ohlc_numeric", "FAIL" if numeric_bad else "PASS", f"bad_values={numeric_bad}"),
        AuditItem(name, "high_ge_open_close_low", "FAIL" if high_bad else "PASS", f"bad_rows={high_bad}"),
        AuditItem(name, "low_le_open_close_high", "FAIL" if low_bad else "PASS", f"bad_rows={low_bad}"),
    ]


def date_gap_audit(name: str, dates: pd.Series) -> list[AuditItem]:
    parsed = pd.to_datetime(dates, errors="coerce").dropna().drop_duplicates().sort_values()
    if parsed.empty:
        return [AuditItem(name, "date_gap_gt_7_calendar_days", "WARN", "no valid dates")]
    gaps = parsed.diff().dt.days.dropna()
    count = int((gaps > 7).sum())
    max_gap = int(gaps.max()) if not gaps.empty else 0
    return [AuditItem(name, "date_gap_gt_7_calendar_days", "WARN" if count else "PASS", f"count={count}; max_gap={max_gap}")]


def nonnegative_check(dataset: str, check: str, values: pd.Series) -> AuditItem:
    bad = int((values.notna() & (values < 0)).sum())
    return AuditItem(dataset, check, "FAIL" if bad else "PASS", f"bad_rows={bad}")


def read_csv_with_encoding(path: Path) -> pd.DataFrame:
    last_exc: Exception | None = None
    for encoding in ENCODINGS:
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False, index_col=False)
        except UnicodeDecodeError as exc:
            last_exc = exc
            continue
    if last_exc is not None:
        raise last_exc
    return pd.DataFrame()


def read_csv_with_detected_encoding(path: Path, nrows: int | None = None) -> tuple[pd.DataFrame, str, str]:
    last_exc: Exception | None = None
    for encoding in ENCODINGS:
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False, nrows=nrows, index_col=False), encoding, ""
        except UnicodeDecodeError as exc:
            last_exc = exc
            continue
        except Exception as exc:
            return pd.DataFrame(), encoding, str(exc)
    return pd.DataFrame(), "", str(last_exc) if last_exc else "unknown read error"


def build_raw_options_file_inventory(opt_dir: Path, processed_options: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    files = sorted(opt_dir.glob("*.csv")) if opt_dir.exists() else []
    for path in files:
        rows.append(raw_options_file_inventory_row(path))
    rows.extend(raw_options_inventory_summary(rows, processed_options))
    return pd.DataFrame(rows)


def raw_options_file_inventory_row(path: Path) -> dict[str, object]:
    base: dict[str, object] = {
        "section": "raw_file",
        "source_file": path.name,
        "file_size": path.stat().st_size if path.exists() else 0,
        "detected_encoding": "",
        "detected_columns": "",
        "row_count": 0,
        "date_min": "",
        "date_max": "",
        "year_min": np.nan,
        "year_max": np.nan,
        "has_2020_rows": False,
        "raw_rows_2020": 0,
        "option_rows_count": 0,
        "normalized_success": False,
        "normalization_error": "",
        "sample_first_date": "",
        "sample_last_date": "",
    }
    if path.suffix.lower() in {".csv", ".txt", ""}:
        sample, encoding, error = read_csv_with_detected_encoding(path, nrows=500)
        base["detected_encoding"] = encoding
        if error:
            base["normalization_error"] = f"read_failed: {error}"
            return base
        sample = normalize_columns(sample)
        base["detected_columns"] = ";".join(map(str, sample.columns.tolist()))
        date_candidates = detect_date_columns(sample)
        contract_cols = detect_contract_columns(sample)
        if not date_candidates:
            base["normalization_error"] = "missing_date_column"
            base["row_count"] = delimited_row_count(path, encoding)
            return base
        counts = delimited_date_and_txo_counts(path, encoding, date_candidates[0], contract_cols[0] if contract_cols else None)
        base["row_count"] = int(counts.get("row_count_raw", 0))
        base["option_rows_count"] = int(counts.get("txo_rows_detected", base["row_count"]))
        base["date_min"] = counts.get("parsed_date_min", "")
        base["date_max"] = counts.get("parsed_date_max", "")
        base["year_min"] = counts.get("parsed_year_min", np.nan)
        base["year_max"] = counts.get("parsed_year_max", np.nan)
        base["raw_rows_2020"] = int(counts.get("raw_rows_2020", 0))
        base["has_2020_rows"] = bool(counts.get("has_2020_rows", False))
        base["sample_first_date"] = base["date_min"]
        base["sample_last_date"] = base["date_max"]
        normalizable, reason = prepare_real_data_normalizable_status(path, sample)
        base["normalized_success"] = bool(normalizable)
        base["normalization_error"] = "" if normalizable else reason
        return base
    df, encoding, error = read_csv_with_detected_encoding(path)
    base["detected_encoding"] = encoding
    if error:
        base["normalization_error"] = f"read_failed: {error}"
        return base
    df = normalize_columns(df)
    base["detected_columns"] = ";".join(map(str, df.columns.tolist()))
    base["row_count"] = int(len(df))
    if df.empty:
        base["normalization_error"] = "empty_file"
        return base
    date_col = first_col(df, ["鈭斗??交?", "?交?", "Date", "date", "?剜???鈭?", "?鈭?"])
    contract_col = first_col(df, ["憟?", "contract", "Contract", "??"])
    if date_col is None:
        base["normalization_error"] = "missing_date_column"
        return base
    work = df.copy()
    if contract_col is not None:
        contract = work[contract_col].astype(str).str.strip().str.upper()
        work = work[contract.eq("TXO")]
        if work.empty:
            base["normalization_error"] = "no_txo_rows"
            return base
    dates = parse_date(work[date_col]).dropna()
    base["option_rows_count"] = int(len(work))
    if dates.empty:
        base["normalization_error"] = "unparseable_dates"
        return base
    date_min = dates.min()
    date_max = dates.max()
    base["date_min"] = str(date_min.date())
    base["date_max"] = str(date_max.date())
    base["year_min"] = int(date_min.year)
    base["year_max"] = int(date_max.year)
    base["raw_rows_2020"] = int((dates.dt.year == 2020).sum())
    base["has_2020_rows"] = bool(base["raw_rows_2020"])
    base["sample_first_date"] = str(dates.iloc[0].date())
    base["sample_last_date"] = str(dates.iloc[-1].date())
    base["normalized_success"] = True
    base["normalization_error"] = ""
    return base


def raw_options_inventory_summary(raw_rows: list[dict[str, object]], processed_options: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    raw = pd.DataFrame(raw_rows)
    processed = processed_options.copy()
    if "date" in processed:
        processed["date"] = pd.to_datetime(processed["date"], errors="coerce")
    raw_files_count = int(len(raw))
    raw_2020_files = int(raw.get("has_2020_rows", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if not raw.empty else 0
    raw_2020_rows = int(pd.to_numeric(raw.get("raw_rows_2020", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not raw.empty else 0
    processed_year_counts = (
        processed["date"].dt.year.value_counts().sort_index()
        if "date" in processed and not processed.empty
        else pd.Series(dtype=int)
    )
    processed_2020_rows = int(processed_year_counts.get(2020, 0))
    status_2020 = "COVERED"
    if raw_2020_rows > 0 and processed_2020_rows == 0:
        status_2020 = "INGESTION_GAP"
    elif raw_2020_rows == 0:
        status_2020 = "RAW_DATA_MISSING"
    rows.append(
        {
            "section": "aggregate_summary",
            "metric": "raw_file_coverage",
            "raw_files_count": raw_files_count,
            "raw_files_covering_2020": raw_2020_files,
            "raw_rows_covering_2020": raw_2020_rows,
            "processed_2020_rows": processed_2020_rows,
            "coverage_status_2020": status_2020,
        }
    )
    if "date" in processed and processed["date"].notna().any():
        rows.append(
            {
                "section": "aggregate_summary",
                "metric": "processed_options_date_range",
                "date_min": str(processed["date"].min().date()),
                "date_max": str(processed["date"].max().date()),
            }
        )
    for year in range(2001, 2026):
        rows.append(
            {
                "section": "aggregate_summary",
                "metric": "processed_options_rows_by_year",
                "year": year,
                "processed_options_rows": int(processed_year_counts.get(year, 0)),
            }
        )
    processed_years = {int(year) for year, count in processed_year_counts.items() if int(count) > 0}
    raw_years = raw_year_set(raw)
    missing_years = [year for year in range(2001, 2026) if year not in processed_years]
    raw_without_processed = sorted(year for year in raw_years if 2001 <= year <= 2025 and year not in processed_years)
    low_row_years = [int(year) for year, count in processed_year_counts.items() if 2001 <= int(year) <= 2025 and int(count) < 1000]
    rows.extend(
        [
            {"section": "aggregate_summary", "metric": "missing_years_2001_2025", "years": ";".join(map(str, missing_years))},
            {"section": "aggregate_summary", "metric": "years_with_raw_data_but_no_processed_data", "years": ";".join(map(str, raw_without_processed))},
            {"section": "aggregate_summary", "metric": "years_with_processed_data_but_low_row_count", "years": ";".join(map(str, low_row_years))},
        ]
    )
    if not raw.empty:
        skipped = raw[~raw["normalized_success"].fillna(False).astype(bool)]
        for row in skipped.itertuples(index=False):
            rows.append(
                {
                    "section": "aggregate_summary",
                    "metric": "source_file_skipped_by_prepare_real_data",
                    "source_file": getattr(row, "source_file", ""),
                    "normalization_error": getattr(row, "normalization_error", ""),
                }
            )
    return rows


def raw_year_set(raw: pd.DataFrame) -> set[int]:
    years: set[int] = set()
    if raw.empty:
        return years
    for row in raw.itertuples(index=False):
        if not bool(getattr(row, "normalized_success", False)):
            continue
        y0 = getattr(row, "year_min", np.nan)
        y1 = getattr(row, "year_max", np.nan)
        if pd.isna(y0) or pd.isna(y1):
            continue
        years.update(range(int(y0), int(y1) + 1))
    return years


def write_raw_options_inventory_markdown(path: Path, inventory: pd.DataFrame) -> None:
    aggregate = inventory[inventory["section"] == "aggregate_summary"] if not inventory.empty and "section" in inventory else pd.DataFrame()
    raw_files = inventory[inventory["section"] == "raw_file"] if not inventory.empty and "section" in inventory else pd.DataFrame()
    coverage = aggregate[aggregate.get("metric", pd.Series(dtype=str)) == "raw_file_coverage"] if not aggregate.empty else pd.DataFrame()
    lines = [
        "# Raw Options File Inventory",
        "",
        "This report audits raw TXO option file coverage and ingestion coverage only. It does not modify raw data, processed data, parsers, prices, formulas, or trades.",
        "",
        "## Summary",
        "",
    ]
    if coverage.empty:
        lines.append("- No coverage summary.")
    else:
        row = coverage.iloc[0]
        lines.append(f"- raw files count: {row.get('raw_files_count')}")
        lines.append(f"- raw files covering 2020: {row.get('raw_files_covering_2020')}")
        lines.append(f"- raw rows covering 2020: {row.get('raw_rows_covering_2020')}")
        lines.append(f"- processed 2020 rows: {row.get('processed_2020_rows')}")
        lines.append(f"- 2020 coverage status: {row.get('coverage_status_2020')}")
    lines.extend(["", "## Skipped Raw Files", ""])
    skipped = raw_files[~raw_files.get("normalized_success", pd.Series(dtype=bool)).fillna(False).astype(bool)] if not raw_files.empty else pd.DataFrame()
    if skipped.empty:
        lines.append("- none")
    else:
        for row in skipped.head(50).itertuples(index=False):
            lines.append(f"- {row.source_file}: {row.normalization_error}")
    missing = aggregate[aggregate.get("metric", pd.Series(dtype=str)) == "missing_years_2001_2025"] if not aggregate.empty else pd.DataFrame()
    if not missing.empty:
        lines.extend(["", "## Missing Years", "", f"- {missing.iloc[0].get('years', '')}"])
    lines.extend(
        [
            "",
            "## Required Limitations",
            "",
            "- This inventory does not fill missing data.",
            "- It does not relax quote quality requirements.",
            "- It does not change ingestion logic.",
            "- It does not run a backtest.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")



def build_raw_options_file_recognition_debug(opt_dir: Path, processed_options: pd.DataFrame) -> pd.DataFrame:
    files = [path for path in sorted(opt_dir.rglob("*")) if path.is_file()] if opt_dir.exists() else []
    rows = [raw_options_file_recognition_row(path) for path in files]
    rows.extend(raw_options_recognition_summary(rows, processed_options))
    return pd.DataFrame(rows)


def raw_options_file_recognition_row(path: Path) -> dict[str, object]:
    extension = path.suffix.lower()
    first_text, detected_encoding = read_first_text(path)
    row: dict[str, object] = {
        "section": "raw_file_recognition",
        "source_file": path.name,
        "full_path": str(path.resolve()),
        "extension": extension if extension else "(none)",
        "file_size": path.stat().st_size if path.exists() else 0,
        "first_300_chars": first_text[:300].replace("\r", " ").replace("\n", " "),
        "detected_encoding": detected_encoding,
        "detected_format": "UNKNOWN",
        "detected_columns": "",
        "row_count_raw": 0,
        "date_column_candidates": "",
        "first_20_date_values": "",
        "parsed_date_min": "",
        "parsed_date_max": "",
        "parsed_year_min": np.nan,
        "parsed_year_max": np.nan,
        "roc_year_detected": False,
        "western_year_detected": False,
        "has_2018_rows": False,
        "has_2019_rows": False,
        "has_2020_rows": False,
        "has_2021_rows": False,
        "has_2022_rows": False,
        "has_2023_rows": False,
        "has_2024_rows": False,
        "has_2018_2024_rows": False,
        "raw_rows_2018": 0,
        "raw_rows_2019": 0,
        "raw_rows_2020": 0,
        "raw_rows_2021": 0,
        "raw_rows_2022": 0,
        "raw_rows_2023": 0,
        "raw_rows_2024": 0,
        "contract_column_detected": False,
        "cp_column_detected": False,
        "strike_column_detected": False,
        "txo_rows_detected": 0,
        "normalizable_by_prepare_real_data": False,
        "skip_reason": "UNKNOWN",
    }
    if int(row["file_size"]) == 0:
        row["skip_reason"] = "EMPTY_FILE"
        return row
    if extension in {".csv", ".txt", ""} and not looks_like_html(first_text):
        return raw_delimited_file_recognition_row(path, row)
    df, detected_format, error = read_tabular_for_recognition(path)
    row["detected_format"] = detected_format
    if error:
        row["skip_reason"] = error
        return row
    if df.empty:
        row["skip_reason"] = "EMPTY_FILE"
        return row
    df = normalize_columns(df)
    row["detected_columns"] = ";".join(map(str, df.columns.tolist()))
    row["row_count_raw"] = int(len(df))
    date_candidates = detect_date_columns(df)
    row["date_column_candidates"] = ";".join(date_candidates)
    parsed_dates = pd.Series(dtype="datetime64[ns]")
    first_values: list[str] = []
    for col in date_candidates:
        values = df[col].dropna().astype(str)
        if not first_values:
            first_values = values.head(20).tolist()
        parsed_dates = pd.concat([parsed_dates, parse_mixed_taifex_dates(values).dropna()], ignore_index=True)
        row["roc_year_detected"] = bool(row["roc_year_detected"]) or detect_roc_year(values)
        row["western_year_detected"] = bool(row["western_year_detected"]) or detect_western_year(values)
    row["first_20_date_values"] = ";".join(first_values)
    if parsed_dates.empty:
        row["skip_reason"] = "NO_DATE_COLUMN" if not date_candidates else "DATE_PARSE_GAP"
        return row
    row["parsed_date_min"] = str(parsed_dates.min().date())
    row["parsed_date_max"] = str(parsed_dates.max().date())
    row["parsed_year_min"] = int(parsed_dates.min().year)
    row["parsed_year_max"] = int(parsed_dates.max().year)
    for year in range(2018, 2025):
        count = int((parsed_dates.dt.year == year).sum())
        row[f"raw_rows_{year}"] = count
        row[f"has_{year}_rows"] = bool(count)
    row["has_2018_2024_rows"] = any(bool(row[f"has_{year}_rows"]) for year in range(2018, 2025))
    contract_cols = detect_contract_columns(df)
    cp_cols = detect_cp_columns(df)
    strike_cols = detect_strike_columns(df)
    row["contract_column_detected"] = bool(contract_cols)
    row["cp_column_detected"] = bool(cp_cols)
    row["strike_column_detected"] = bool(strike_cols)
    row["txo_rows_detected"] = count_txo_rows(df, contract_cols)
    normalizable, reason = prepare_real_data_normalizable_status(path, df)
    row["normalizable_by_prepare_real_data"] = bool(normalizable)
    if normalizable:
        row["skip_reason"] = "OK_RECOGNIZED"
    elif not contract_cols:
        row["skip_reason"] = "TXO_COLUMN_PARSE_GAP"
    elif int(row["txo_rows_detected"]) == 0:
        row["skip_reason"] = "NO_TXO_ROWS"
    elif not cp_cols or not strike_cols:
        row["skip_reason"] = "COLUMN_MAPPING_GAP"
    else:
        row["skip_reason"] = reason or "COLUMN_MAPPING_GAP"
    return row


def raw_delimited_file_recognition_row(path: Path, row: dict[str, object]) -> dict[str, object]:
    sample, encoding, error = read_csv_with_detected_encoding(path, nrows=500)
    row["detected_encoding"] = encoding or row.get("detected_encoding", "")
    row["detected_format"] = "CSV"
    if error:
        row["skip_reason"] = "ENCODING_FAILED"
        return row
    sample = normalize_columns(sample)
    row["detected_columns"] = ";".join(map(str, sample.columns.tolist()))
    date_candidates = detect_date_columns(sample)
    contract_cols = detect_contract_columns(sample)
    cp_cols = detect_cp_columns(sample)
    strike_cols = detect_strike_columns(sample)
    row["date_column_candidates"] = ";".join(date_candidates)
    row["contract_column_detected"] = bool(contract_cols)
    row["cp_column_detected"] = bool(cp_cols)
    row["strike_column_detected"] = bool(strike_cols)
    if date_candidates:
        first_values = sample[date_candidates[0]].dropna().astype(str).head(20)
        row["first_20_date_values"] = ";".join(first_values.tolist())
        row["roc_year_detected"] = detect_roc_year(first_values)
        row["western_year_detected"] = detect_western_year(first_values)
    if not date_candidates:
        row["skip_reason"] = "NO_DATE_COLUMN"
        row["row_count_raw"] = delimited_row_count(path, encoding)
        return row
    row.update(delimited_date_and_txo_counts(path, encoding, date_candidates[0], contract_cols[0] if contract_cols else None))
    normalizable, reason = prepare_real_data_normalizable_status(path, sample)
    row["normalizable_by_prepare_real_data"] = bool(normalizable)
    if normalizable:
        row["skip_reason"] = "OK_RECOGNIZED"
    elif not contract_cols:
        row["skip_reason"] = "TXO_COLUMN_PARSE_GAP"
    elif int(row["txo_rows_detected"]) == 0:
        row["skip_reason"] = "NO_TXO_ROWS"
    elif not cp_cols or not strike_cols:
        row["skip_reason"] = "COLUMN_MAPPING_GAP"
    else:
        row["skip_reason"] = reason or "COLUMN_MAPPING_GAP"
    return row


def delimited_row_count(path: Path, encoding: str) -> int:
    try:
        with path.open("r", encoding=encoding or "utf-8-sig", errors="ignore") as handle:
            return max(0, sum(1 for _ in handle) - 1)
    except Exception:
        return 0


def delimited_date_and_txo_counts(path: Path, encoding: str, date_col: str, contract_col: str | None) -> dict[str, object]:
    row_count = delimited_row_count(path, encoding)
    txo_count = 0
    parsed_dates: list[pd.Timestamp] = []
    try:
        with path.open("r", encoding=encoding or "utf-8-sig", errors="ignore") as handle:
            header_line = handle.readline()
            header = [part.replace("\ufeff", "").strip() for part in header_line.strip().split(",")]
            date_idx = header.index(date_col)
            contract_idx = header.index(contract_col) if contract_col and contract_col in header else None
            sample_lines = []
            for _, line in zip(range(2000), handle):
                sample_lines.append(line)
            tail_lines = tail_text_lines(path, encoding, 2000)
            sampled = sample_lines + tail_lines
            for line in sampled:
                parts = line.rstrip("\n\r").split(",")
                if date_idx < len(parts):
                    parsed = parse_mixed_taifex_date_value(parts[date_idx])
                    if pd.notna(parsed):
                        parsed_dates.append(parsed)
                if contract_idx is not None and contract_idx < len(parts) and parts[contract_idx].strip().upper() == "TXO":
                    txo_count += 1
    except Exception:
        return {"row_count_raw": row_count, "skip_reason": "DATE_PARSE_GAP"}
    out: dict[str, object] = {"row_count_raw": int(row_count), "txo_rows_detected": int(txo_count)}
    if not parsed_dates:
        out["skip_reason"] = "DATE_PARSE_GAP"
        return out
    parsed_series = pd.Series(parsed_dates)
    out["parsed_date_min"] = str(parsed_series.min().date())
    out["parsed_date_max"] = str(parsed_series.max().date())
    out["parsed_year_min"] = int(parsed_series.min().year)
    out["parsed_year_max"] = int(parsed_series.max().year)
    for year in range(2018, 2025):
        if int(parsed_series.min().year) == int(parsed_series.max().year) == year:
            count = int(row_count)
        else:
            count = int((parsed_series.dt.year == year).sum())
        out[f"raw_rows_{year}"] = count
        out[f"has_{year}_rows"] = bool(count)
    out["has_2018_2024_rows"] = any(bool(out[f"has_{year}_rows"]) for year in range(2018, 2025))
    return out


def tail_text_lines(path: Path, encoding: str, n: int) -> list[str]:
    try:
        with path.open("r", encoding=encoding or "utf-8-sig", errors="ignore") as handle:
            from collections import deque

            return list(deque(handle, maxlen=n))
    except Exception:
        return []


def read_first_text(path: Path, size: int = 4096) -> tuple[str, str]:
    raw = path.read_bytes()[:size]
    for encoding in ENCODINGS:
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace"), "binary_or_unknown"


def read_tabular_for_recognition(path: Path) -> tuple[pd.DataFrame, str, str]:
    ext = path.suffix.lower()
    text, _ = read_first_text(path, size=500_000)
    if ext in {".csv", ".txt", ""}:
        if looks_like_html(text):
            tables = read_html_tables_for_recognition(text)
            return (tables[0], "EXCEL_HTML", "") if tables else (pd.DataFrame(), "HTML", "HTML_NOT_CSV")
        df, _, error = read_csv_with_detected_encoding(path)
        return (df, "CSV", "") if not error else (pd.DataFrame(), "UNKNOWN", "ENCODING_FAILED")
    if ext in {".htm", ".html"}:
        tables = read_html_tables_for_recognition(text)
        return (tables[0], "HTML", "") if tables else (pd.DataFrame(), "HTML", "HTML_NOT_CSV")
    if ext in {".xls", ".xlsx"}:
        try:
            return pd.read_excel(path), "EXCEL", ""
        except Exception:
            if looks_like_html(text):
                tables = read_html_tables_for_recognition(text)
                return (tables[0], "EXCEL_HTML", "") if tables else (pd.DataFrame(), "EXCEL_HTML", "HTML_NOT_CSV")
            return pd.DataFrame(), "EXCEL", "UNSUPPORTED_EXTENSION"
    return pd.DataFrame(), "UNKNOWN", "UNSUPPORTED_EXTENSION"


def looks_like_html(text: str) -> bool:
    sample = text[:500].lower()
    return "<html" in sample or "<table" in sample or "<!doctype html" in sample


def read_html_tables_for_recognition(text: str) -> list[pd.DataFrame]:
    try:
        return pd.read_html(text)
    except Exception:
        return []


def detect_date_columns(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for col in df.columns:
        name = str(col).lower()
        values = df[col].dropna().astype(str).head(100)
        parsed = parse_mixed_taifex_dates(values)
        if any(token in name for token in ["date", "\u65e5\u671f", "\u4ea4\u6613\u65e5"]) or parsed.notna().sum() >= max(3, min(10, len(values)) // 2):
            cols.append(str(col))
    return cols


def parse_mixed_taifex_dates(values: pd.Series) -> pd.Series:
    return values.apply(parse_mixed_taifex_date_value)


def parse_mixed_taifex_date_value(value: object) -> pd.Timestamp | pd.NaT:
    text = str(value).strip().strip('"')
    if not text or text.lower() == "nan":
        return pd.NaT
    match = re.fullmatch(r"(\d{2,4})[/-](\d{1,2})[/-](\d{1,2})", text)
    if match:
        year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
        if 80 <= year <= 150:
            year += 1911
        return safe_timestamp(year, month, day)
    match = re.fullmatch(r"(\d{7,8})", text)
    if match:
        digits = match.group(1)
        year = int(digits[:3]) + 1911 if len(digits) == 7 else int(digits[:4])
        month = int(digits[-4:-2])
        day = int(digits[-2:])
        return safe_timestamp(year, month, day)
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed) or parsed.year < 1900 or parsed.year > 2100:
        return pd.NaT
    return parsed


def safe_timestamp(year: int, month: int, day: int) -> pd.Timestamp | pd.NaT:
    if year < 1900 or year > 2100:
        return pd.NaT
    try:
        return pd.Timestamp(year=year, month=month, day=day)
    except ValueError:
        return pd.NaT


def detect_roc_year(values: pd.Series) -> bool:
    return values.astype(str).str.match(r"^\s*(10[7-9]|11[0-3])([/-]|\d{4})").any()


def detect_western_year(values: pd.Series) -> bool:
    return values.astype(str).str.match(r"^\s*20(18|19|20|21|22|23|24)([/-]|\d{4})").any()


def detect_contract_columns(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for col in df.columns:
        name = str(col).lower()
        values = df[col].dropna().astype(str).str.upper().head(500)
        if any(token in name for token in ["contract", "\u5951\u7d04", "\u5546\u54c1"]) or values.isin(TXO_CONTRACT_ALIASES).any():
            cols.append(str(col))
    return cols


def detect_cp_columns(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for col in df.columns:
        name = str(col).lower()
        values = df[col].dropna().astype(str).str.upper().head(500)
        if any(token in name for token in ["\u8cb7\u8ce3\u6b0a", "cp", "call", "put"]) or values.isin(["C", "P", "CALL", "PUT"]).any() or values.str.contains("\u8cb7\u6b0a|\u8ce3\u6b0a|CALL|PUT", regex=True).any():
            cols.append(str(col))
    return cols


def detect_strike_columns(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for col in df.columns:
        name = str(col).lower()
        numeric = pd.to_numeric(df[col], errors="coerce")
        if any(token in name for token in ["\u5c65\u7d04", "strike"]) or numeric.between(1000, 50000).sum() >= max(3, min(20, len(df)) // 2):
            cols.append(str(col))
    return cols


def count_txo_rows(df: pd.DataFrame, contract_cols: list[str]) -> int:
    if not contract_cols:
        return 0
    mask = pd.Series(False, index=df.index)
    for col in contract_cols:
        mask |= df[col].astype(str).str.strip().str.upper().eq("TXO")
    return int(mask.sum())


def prepare_real_data_normalizable_status(path: Path, df: pd.DataFrame) -> tuple[bool, str]:
    if path.suffix.lower() != ".csv":
        return False, "UNSUPPORTED_EXTENSION"
    date_col = first_col(df, ["?剜???鈭?", "?鈭?"])
    contract_col = first_col(df, ["??"])
    if date_col is None or contract_col is None:
        return False, "COLUMN_MAPPING_GAP"
    txo = df[df[contract_col].astype(str).str.strip().eq("TXO")]
    if txo.empty:
        return False, "NO_TXO_ROWS"
    if parse_date(txo[date_col]).dropna().empty:
        return False, "DATE_PARSE_GAP"
    return True, ""


def raw_options_recognition_summary(raw_rows: list[dict[str, object]], processed_options: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    raw = pd.DataFrame(raw_rows)
    processed = processed_options.copy()
    if "date" in processed:
        processed["date"] = pd.to_datetime(processed["date"], errors="coerce")
    processed_counts = processed["date"].dt.year.value_counts().sort_index() if "date" in processed and not processed.empty else pd.Series(dtype=int)
    rows.append({"section": "aggregate_summary", "metric": "raw_files_count", "value": int(len(raw))})
    if raw.empty:
        for year in range(2018, 2025):
            processed_count = int(processed_counts.get(year, 0))
            status = "RAW_DATA_MISSING" if processed_count == 0 else "COVERED"
            rows.append({"section": "aggregate_summary", "metric": "year_coverage_status", "year": year, "raw_rows": 0, "processed_rows": processed_count, "coverage_status": status})
        return rows
    rows.append({"section": "aggregate_summary", "metric": "files_with_2018_2024_rows", "value": int(raw["has_2018_2024_rows"].fillna(False).astype(bool).sum())})
    rows.append({"section": "aggregate_summary", "metric": "files_with_2020_rows", "value": int(raw["has_2020_rows"].fillna(False).astype(bool).sum())})
    for ext, count in raw["extension"].fillna("(none)").astype(str).value_counts().items():
        rows.append({"section": "aggregate_summary", "metric": "extension_distribution", "extension": ext, "count": int(count)})
    for reason, count in raw["skip_reason"].fillna("UNKNOWN").astype(str).value_counts().items():
        rows.append({"section": "aggregate_summary", "metric": "skip_reason_distribution", "skip_reason": reason, "count": int(count)})
    for year in range(2018, 2025):
        raw_count = int(pd.to_numeric(raw.get(f"raw_rows_{year}", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        processed_count = int(processed_counts.get(year, 0))
        status = "INGESTION_RECOGNITION_GAP" if raw_count > 0 and processed_count == 0 else "RAW_DATA_MISSING" if raw_count == 0 else "COVERED"
        rows.append({"section": "aggregate_summary", "metric": "year_coverage_status", "year": year, "raw_rows": raw_count, "processed_rows": processed_count, "coverage_status": status})
    gap_files = raw[raw["has_2018_2024_rows"].fillna(False).astype(bool) & ~raw["normalizable_by_prepare_real_data"].fillna(False).astype(bool)]
    if not gap_files.empty:
        top_reason = gap_files["skip_reason"].fillna("UNKNOWN").astype(str).value_counts().idxmax()
        rows.append({"section": "aggregate_summary", "metric": "most_likely_prepare_real_data_gap_reason", "skip_reason": top_reason, "count": int((gap_files["skip_reason"].astype(str) == top_reason).sum())})
    return rows


def write_raw_options_recognition_debug_markdown(path: Path, debug: pd.DataFrame) -> None:
    raw = debug[debug["section"] == "raw_file_recognition"] if not debug.empty and "section" in debug else pd.DataFrame()
    agg = debug[debug["section"] == "aggregate_summary"] if not debug.empty and "section" in debug else pd.DataFrame()
    lines = ["# Raw Options File Recognition Debug", "", "This report recursively scans raw option files for recognition gaps. It does not modify raw data, processed data, parsers, formulas, strategies, or trades.", "", "## Summary", ""]
    for metric in ["raw_files_count", "files_with_2018_2024_rows", "files_with_2020_rows"]:
        item = agg[agg.get("metric", pd.Series(dtype=str)) == metric] if not agg.empty else pd.DataFrame()
        lines.append(f"- {metric}: {item.iloc[0].get('value', '') if not item.empty else ''}")
    lines.extend(["", "## Skip Reason Distribution", ""])
    for row in agg[agg.get("metric", pd.Series(dtype=str)) == "skip_reason_distribution"].itertuples(index=False) if not agg.empty else []:
        lines.append(f"- {row.skip_reason}: {row.count}")
    lines.extend(["", "## Year Coverage Status", ""])
    for row in agg[agg.get("metric", pd.Series(dtype=str)) == "year_coverage_status"].itertuples(index=False) if not agg.empty else []:
        lines.append(f"- {row.year}: raw_rows={row.raw_rows}, processed_rows={row.processed_rows}, status={row.coverage_status}")
    gap = agg[agg.get("metric", pd.Series(dtype=str)) == "most_likely_prepare_real_data_gap_reason"] if not agg.empty else pd.DataFrame()
    if not gap.empty:
        row = gap.iloc[0]
        lines.extend(["", "## Most Likely Recognition Gap", "", f"- {row.get('skip_reason')}: {row.get('count')} files"])
    files_2020 = raw[raw.get("has_2020_rows", pd.Series(dtype=bool)).fillna(False).astype(bool)] if not raw.empty else pd.DataFrame()
    lines.extend(["", "## Files With 2020 Rows", ""])
    if files_2020.empty:
        lines.append("- none")
    else:
        for row in files_2020.head(100).itertuples(index=False):
            lines.append(f"- {row.source_file}: skip_reason={row.skip_reason}, parsed_range={row.parsed_date_min}..{row.parsed_date_max}")
    lines.extend(["", "## Required Limitations", "", "- This debug report does not download data.", "- It does not fill or repair missing data.", "- It does not change parser behavior.", "- It does not run a backtest."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(col).replace("\ufeff", "").strip() for col in out.columns]
    unnamed = [col for col in out.columns if str(col).startswith("Unnamed:")]
    if unnamed:
        out = out.drop(columns=unnamed)
    return out


def first_col(df: pd.DataFrame, names: Iterable[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def col_or_nan(df: pd.DataFrame, names: Iterable[str]) -> pd.Series:
    col = first_col(df, names)
    if col is None:
        return pd.Series(np.nan, index=df.index)
    return df[col]


def to_number(values: pd.Series) -> pd.Series:
    cleaned = values.astype(str).str.strip().replace({"": np.nan, "-": np.nan, "--": np.nan, "nan": np.nan})
    cleaned = cleaned.str.replace(",", "", regex=False).str.replace("%", "", regex=False)
    return pd.to_numeric(cleaned, errors="coerce")


def parse_date(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce")


def normalize_cp(value: object) -> str | float:
    text = str(value).strip().upper()
    mapping = {"鞎瑟?": "C", "鞈??": "P", "\u8cb7\u6b0a": "C", "\u8ce3\u6b0a": "P", "CALL": "C", "PUT": "P", "C": "C", "P": "P"}
    return mapping.get(text, np.nan)


def expiry_sort_key(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.extract(r"(\d{6})", expand=False)
    return pd.to_numeric(text, errors="coerce").fillna(999999)


def load_expiry_calendar(path: Path) -> ExpiryCalendar:
    if not path.exists():
        return ExpiryCalendar(path=path, exists=False, overrides={}, warnings=())
    warnings: list[str] = []
    overrides: dict[str, pd.Timestamp] = {}
    try:
        df = read_csv_with_encoding(path)
    except Exception as exc:
        return ExpiryCalendar(path=path, exists=True, overrides={}, warnings=(f"read_failed: {exc}",))
    df = normalize_columns(df)
    missing = [col for col in ["expiry_raw", "actual_expiry"] if col not in df.columns]
    if missing:
        return ExpiryCalendar(path=path, exists=True, overrides={}, warnings=(f"missing_columns: {','.join(missing)}",))
    for row in df.itertuples(index=False):
        raw = str(getattr(row, "expiry_raw")).strip().upper()
        actual = pd.to_datetime(getattr(row, "actual_expiry"), errors="coerce")
        if not raw or raw == "NAN":
            warnings.append("blank_expiry_raw")
            continue
        if pd.isna(actual):
            warnings.append(f"{raw}: invalid actual_expiry")
            continue
        if expiry_raw_pattern(raw) == "unrecognized":
            warnings.append(f"{raw}: invalid expiry_raw format")
            continue
        reasonable_floor = expiry_month_floor(raw)
        if pd.notna(reasonable_floor) and actual < reasonable_floor:
            warnings.append(f"{raw}: actual_expiry before contract month")
            continue
        overrides[raw] = pd.Timestamp(actual).normalize()
    return ExpiryCalendar(path=path, exists=True, overrides=overrides, warnings=tuple(warnings))


def parse_taifex_expiry(value: object, overrides: dict[str, pd.Timestamp] | None = None) -> pd.Timestamp | pd.NaT:
    text = str(value).strip().upper()
    if overrides and text in overrides:
        return overrides[text]
    return parse_expiry_by_rule(text)


def parse_expiry_by_rule(value: object) -> pd.Timestamp | pd.NaT:
    text = str(value).strip().upper()
    match = re.fullmatch(r"(\d{4})(\d{2})(?:([WF])([1-5]))?", text)
    if match is None:
        return pd.NaT
    year = int(match.group(1))
    month = int(match.group(2))
    if month < 1 or month > 12:
        return pd.NaT
    marker = match.group(3)
    nth = int(match.group(4)) if match.group(4) else 3
    weekday = calendar.FRIDAY if marker == "F" else calendar.WEDNESDAY
    day = nth_weekday(year, month, weekday, nth)
    return pd.Timestamp(year=year, month=month, day=day) if day is not None else pd.NaT


def expiry_month_floor(value: object) -> pd.Timestamp | pd.NaT:
    text = str(value).strip().upper()
    match = re.fullmatch(r"(\d{4})(\d{2})(?:([WF])([1-5]))?", text)
    if match is None:
        return pd.NaT
    year = int(match.group(1))
    month = int(match.group(2))
    if month < 1 or month > 12:
        return pd.NaT
    return pd.Timestamp(year=year, month=month, day=1)


def nth_weekday(year: int, month: int, weekday: int, nth: int) -> int | None:
    count = 0
    for day in range(1, calendar.monthrange(year, month)[1] + 1):
        if calendar.weekday(year, month, day) == weekday:
            count += 1
            if count == nth:
                return day
    return None


def concat_or_empty(frames: list[pd.DataFrame], columns: list[str]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)


def spread_pct(df: pd.DataFrame) -> pd.Series:
    mid = (df["ask"] + df["bid"]) / 2.0
    return (df["ask"] - df["bid"]) / mid.replace(0, np.nan)


def write_debug_outputs(report_dir: Path, options: pd.DataFrame) -> dict[str, int]:
    bad_expiry = bad_expiry_rows(options)
    bad_bid_ask = bad_bid_ask_rows(options)
    extreme_spread = extreme_spread_rows(options)

    bad_expiry_cols = ["date", "contract", "expiry_raw", "expiry", "dte", "cp", "strike", "source_file"]
    bad_bid_ask_cols = [
        "date",
        "contract",
        "expiry_raw",
        "expiry",
        "dte",
        "cp",
        "strike",
        "bid",
        "ask",
        "close",
        "settlement_price",
        "volume",
        "open_interest",
        "session",
        "halt_flag",
        "source_file",
    ]
    extreme_cols = bad_bid_ask_cols[:-1] + ["spread_pct", "source_file"]

    bad_expiry[existing_cols(bad_expiry, bad_expiry_cols)].to_csv(report_dir / "bad_expiry_rows.csv", index=False)
    bad_bid_ask[existing_cols(bad_bid_ask, bad_bid_ask_cols)].to_csv(report_dir / "bad_bid_ask_rows.csv", index=False)
    extreme_spread.head(5000)[existing_cols(extreme_spread, extreme_cols)].to_csv(report_dir / "extreme_spread_rows_sample.csv", index=False)
    return {
        "bad_expiry_rows": int(len(bad_expiry)),
        "bad_bid_ask_rows": int(len(bad_bid_ask)),
        "extreme_spread_rows_sample": int(min(len(extreme_spread), 5000)),
    }


def write_quote_quality_reports(report_dir: Path, options: pd.DataFrame) -> None:
    rows: list[dict[str, str | int | float]] = []
    total = len(options)
    tradable = int(options["is_tradable_quote"].fillna(False).sum()) if "is_tradable_quote" in options else 0
    rows.append(
        {
            "section": "summary",
            "bucket": "is_tradable_quote_ratio",
            "quote_quality_status": "ALL",
            "count": tradable,
            "total": total,
            "ratio": tradable / total if total else np.nan,
        }
    )
    rows.extend(quote_quality_distribution(options, "overall", pd.Series("ALL", index=options.index)))
    if "date" in options:
        years = pd.to_datetime(options["date"], errors="coerce").dt.year.astype("Int64").astype(str)
        rows.extend(quote_quality_distribution(options, "by_year", years))
    if "source_file" in options:
        rows.extend(quote_quality_distribution(options, "by_source_file", options["source_file"].astype(str)))
    if "dte" in options:
        dte_bucket = pd.cut(
            options["dte"],
            bins=[-np.inf, 0, 7, 14, 30, 60, 120, np.inf],
            labels=["<0", "0-7", "8-14", "15-30", "31-60", "61-120", ">120"],
        ).astype(str)
        rows.extend(quote_quality_distribution(options, "by_dte_bucket", dte_bucket))
    if "volume" in options:
        volume_bucket = pd.cut(
            options["volume"].fillna(-1),
            bins=[-np.inf, 0, 10, 50, 100, 500, np.inf],
            labels=["missing_or_0", "1-10", "11-50", "51-100", "101-500", ">500"],
        ).astype(str)
        rows.extend(quote_quality_distribution(options, "by_volume_bucket", volume_bucket))
    if "open_interest" in options:
        oi_bucket = pd.cut(
            options["open_interest"].fillna(-1),
            bins=[-np.inf, 0, 10, 50, 100, 500, 1000, np.inf],
            labels=["missing_or_0", "1-10", "11-50", "51-100", "101-500", "501-1000", ">1000"],
        ).astype(str)
        rows.extend(quote_quality_distribution(options, "by_open_interest_bucket", oi_bucket))

    report = pd.DataFrame(rows)
    report.to_csv(report_dir / "options_quote_quality.csv", index=False)
    write_quote_quality_markdown(report_dir / "options_quote_quality.md", options, report)


def quote_quality_distribution(options: pd.DataFrame, section: str, buckets: pd.Series) -> list[dict[str, str | int | float]]:
    if options.empty or "quote_quality_status" not in options:
        return []
    work = pd.DataFrame({"bucket": buckets.fillna("nan").astype(str), "quote_quality_status": options["quote_quality_status"].astype(str)})
    grouped = work.groupby(["bucket", "quote_quality_status"], dropna=False).size().reset_index(name="count")
    totals = work.groupby("bucket", dropna=False).size().rename("total")
    grouped = grouped.merge(totals, on="bucket", how="left")
    grouped["ratio"] = grouped["count"] / grouped["total"]
    grouped.insert(0, "section", section)
    return grouped.to_dict("records")


def write_quote_quality_markdown(path: Path, options: pd.DataFrame, report: pd.DataFrame) -> None:
    total = len(options)
    counts = options["quote_quality_status"].value_counts(dropna=False).to_dict() if "quote_quality_status" in options else {}
    tradable = int(options["is_tradable_quote"].fillna(False).sum()) if "is_tradable_quote" in options else 0
    lines = [
        "# Options Quote Quality",
        "",
        "This report only labels quote quality. It does not fix bid/ask, delete rows, estimate missing prices, or change strategy logic.",
        "",
        "## Summary",
        "",
        f"- total rows: {total}",
        f"- is_tradable_quote rows: {tradable}",
        f"- is_tradable_quote ratio: {tradable / total:.6f}" if total else "- is_tradable_quote ratio: nan",
        "",
        "## quote_quality_status Distribution",
        "",
    ]
    if counts:
        for status, count in counts.items():
            lines.append(f"- {status}: {int(count)}")
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Top Non-VALID By Source File",
            "",
        ]
    )
    non_valid = report[(report["section"] == "by_source_file") & (report["quote_quality_status"] != "VALID")] if not report.empty else pd.DataFrame()
    if non_valid.empty:
        lines.append("- none")
    else:
        top = non_valid.sort_values("count", ascending=False).head(20)
        for row in top.itertuples(index=False):
            lines.append(f"- {row.bucket} / {row.quote_quality_status}: {int(row.count)}")
    lines.extend(
        [
            "",
            "## Bucket Reports",
            "",
            "- Full distributions are in `reports/options_quote_quality.csv`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def bad_expiry_rows(options: pd.DataFrame) -> pd.DataFrame:
    if options.empty:
        return pd.DataFrame(columns=OPTION_COLUMNS + ["expiry_raw"])
    date = pd.to_datetime(options["date"], errors="coerce")
    expiry = pd.to_datetime(options["expiry"], errors="coerce")
    mask = ((expiry.notna()) & (date.notna()) & (expiry < date)) | (options["dte"].notna() & (options["dte"] < 0))
    return options.loc[mask].copy()


def bad_bid_ask_rows(options: pd.DataFrame) -> pd.DataFrame:
    if options.empty:
        return pd.DataFrame(columns=OPTION_COLUMNS + ["expiry_raw"])
    mask = options["bid"].notna() & options["ask"].notna() & (options["bid"] > options["ask"])
    return options.loc[mask].copy()


def extreme_spread_rows(options: pd.DataFrame) -> pd.DataFrame:
    if options.empty:
        out = pd.DataFrame(columns=OPTION_COLUMNS + ["expiry_raw", "spread_pct"])
        return out
    out = options.copy()
    out["spread_pct"] = spread_pct(out)
    return out.loc[out["spread_pct"] > 1.0].copy()


def existing_cols(df: pd.DataFrame, columns: list[str]) -> list[str]:
    for col in columns:
        if col not in df.columns:
            df[col] = np.nan
    return columns


def date_range_detail(df: pd.DataFrame) -> str:
    if "date" not in df or df.empty:
        return "no dates"
    dates = pd.to_datetime(df["date"], errors="coerce").dropna()
    if dates.empty:
        return "no valid dates"
    return f"{dates.min().date()} to {dates.max().date()}"


def distribution(df: pd.DataFrame, column: str) -> str:
    if df.empty or column not in df:
        return "none"
    counts = df[column].astype(str).value_counts(dropna=False).head(20)
    return "; ".join(f"{idx}={val}" for idx, val in counts.items())


def write_audit_markdown(path: Path, audit: pd.DataFrame, market: pd.DataFrame, options: pd.DataFrame) -> None:
    counts = audit["status"].value_counts().to_dict() if not audit.empty else {}
    diagnostics = build_debug_summary(options)
    lines = [
        "# Data Cleaning Audit",
        "",
        "This report audits raw-to-processed data cleaning only. It does not run a backtest, optimize parameters, smooth prices, estimate IV/delta, or change strategy logic.",
        "",
        "## Row Counts",
        "",
        f"- market rows: {len(market)}",
        f"- options rows: {len(options)}",
        "",
        "## Status Counts",
        "",
    ]
    for status in ["PASS", "WARN", "FAIL"]:
        lines.append(f"- {status}: {int(counts.get(status, 0))}")
    lines.extend(["", "## WARN / FAIL Items", ""])
    flagged = audit[audit["status"].isin(["WARN", "FAIL"])] if not audit.empty else pd.DataFrame()
    if flagged.empty:
        lines.append("- None")
    else:
        for row in flagged.itertuples(index=False):
            lines.append(f"- {row.status} `{row.dataset}.{row.check}`: {row.detail}")
    lines.extend(
        [
            "",
            "## Debug Diagnostics",
            "",
            f"- override used count: {diagnostics['override_used_count']}",
            f"- bad expiry count before override: {diagnostics['bad_expiry_before_override']}",
            f"- bad expiry count after override: {diagnostics['bad_expiry_count']}",
            f"- bad expiry count: {diagnostics['bad_expiry_count']}",
            f"- bad expiry first 20 expiry_raw examples: {diagnostics['bad_expiry_examples']}",
            f"- bad expiry by expiry_raw pattern: {diagnostics['bad_expiry_by_pattern']}",
            f"- bad expiry by year: {diagnostics['bad_expiry_by_year']}",
            f"- bad expiry by source_file: {diagnostics['bad_expiry_by_source_file']}",
            f"- bad bid/ask by source_file: {diagnostics['bad_bid_ask_by_source_file']}",
            f"- bad bid/ask by session: {diagnostics['bad_bid_ask_by_session']}",
            f"- extreme spread by DTE bucket: {diagnostics['extreme_spread_by_dte_bucket']}",
            f"- extreme spread by volume bucket: {diagnostics['extreme_spread_by_volume_bucket']}",
            f"- ask = 0 count: {diagnostics['ask_zero_count']}",
            f"- bid = 0 count: {diagnostics['bid_zero_count']}",
            f"- quote_quality_status distribution: {diagnostics['quote_quality_distribution']}",
            f"- is_tradable_quote ratio: {diagnostics['is_tradable_quote_ratio']}",
        ]
    )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- VIX remains blank unless supplied elsewhere.",
            "- IV and delta are intentionally not estimated in this script.",
            "- Abnormal rows are retained and only marked in the audit.",
            "- Official TAIFEX daily futures are preferred over CrazyIndicator minute-derived daily bars on overlapping dates.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_debug_summary(options: pd.DataFrame) -> dict[str, str | int]:
    bad_expiry = bad_expiry_rows(options)
    bad_bid_ask = bad_bid_ask_rows(options)
    extreme = extreme_spread_rows(options)
    if not extreme.empty:
        dte_bucket = pd.cut(
            extreme["dte"],
            bins=[-np.inf, 0, 7, 14, 30, 60, 120, np.inf],
            labels=["<0", "0-7", "8-14", "15-30", "31-60", "61-120", ">120"],
        )
        volume_bucket = pd.cut(
            extreme["volume"].fillna(-1),
            bins=[-np.inf, 0, 10, 50, 100, 500, np.inf],
            labels=["missing_or_0", "1-10", "11-50", "51-100", "101-500", ">500"],
        )
        dte_detail = value_counts_detail(dte_bucket)
        volume_detail = value_counts_detail(volume_bucket)
    else:
        dte_detail = "none"
        volume_detail = "none"
    examples = (
        bad_expiry["expiry_raw"].dropna().astype(str).drop_duplicates().head(20).tolist()
        if "expiry_raw" in bad_expiry
        else []
    )
    return {
        "override_used_count": int(options["override_used"].fillna(False).sum()) if "override_used" in options else 0,
        "bad_expiry_before_override": bad_expiry_count(options, "rule_expiry", "rule_dte"),
        "bad_expiry_count": int(len(bad_expiry)),
        "bad_expiry_examples": "; ".join(examples) if examples else "none",
        "bad_expiry_by_pattern": value_counts_detail(bad_expiry["expiry_raw"].map(expiry_raw_pattern)) if "expiry_raw" in bad_expiry else "none",
        "bad_expiry_by_year": value_counts_detail(pd.to_datetime(bad_expiry["date"], errors="coerce").dt.year) if "date" in bad_expiry else "none",
        "bad_expiry_by_source_file": value_counts_detail(bad_expiry["source_file"]) if "source_file" in bad_expiry else "none",
        "bad_bid_ask_by_source_file": value_counts_detail(bad_bid_ask["source_file"]) if "source_file" in bad_bid_ask else "none",
        "bad_bid_ask_by_session": value_counts_detail(bad_bid_ask["session"]) if "session" in bad_bid_ask else "none",
        "extreme_spread_by_dte_bucket": dte_detail,
        "extreme_spread_by_volume_bucket": volume_detail,
        "ask_zero_count": int((options["ask"].fillna(np.nan) == 0).sum()) if "ask" in options else 0,
        "bid_zero_count": int((options["bid"].fillna(np.nan) == 0).sum()) if "bid" in options else 0,
        "quote_quality_distribution": value_counts_detail(options["quote_quality_status"]) if "quote_quality_status" in options else "none",
        "is_tradable_quote_ratio": (
            f"{float(options['is_tradable_quote'].fillna(False).mean()):.6f}" if "is_tradable_quote" in options and len(options) else "nan"
        ),
    }


def value_counts_detail(values: pd.Series) -> str:
    counts = values.astype(str).value_counts(dropna=False).head(20)
    if counts.empty:
        return "none"
    return "; ".join(f"{idx}={val}" for idx, val in counts.items())


def expiry_raw_pattern(value: object) -> str:
    text = str(value).strip().upper()
    if re.fullmatch(r"\d{6}", text):
        return "YYYYMM monthly"
    if re.fullmatch(r"\d{6}W[1-5]", text):
        return "YYYYMMWn weekly_wed"
    if re.fullmatch(r"\d{6}F[1-5]", text):
        return "YYYYMMFn weekly_fri"
    if re.fullmatch(r"\d{6}[A-Z]\d+", text):
        return "YYYYMM other_weekly_marker"
    return "unrecognized"


if __name__ == "__main__":
    main()


