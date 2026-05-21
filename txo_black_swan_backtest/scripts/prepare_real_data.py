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
    "data_source",
    "source_file",
]


@dataclass(frozen=True)
class AuditItem:
    dataset: str
    check: str
    status: str
    detail: str


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    report_dir = Path(args.report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    txf_market = load_txf_1min(raw_dir / "taifex" / "txf")
    official_market = load_official_futures(raw_dir / "taifex" / "fut")
    market = merge_market_sources(txf_market, official_market)
    options = load_official_options(raw_dir / "taifex" / "opt")

    market.to_csv(out_dir / "market.csv", index=False)
    options[[col for col in OPTION_COLUMNS if col in options.columns]].to_csv(out_dir / "options.csv", index=False)

    audit = build_audit(market, options, txf_market, official_market)
    audit_df = pd.DataFrame([item.__dict__ for item in audit])
    audit_df.to_csv(report_dir / "data_cleaning_audit.csv", index=False)
    debug_counts = write_debug_outputs(report_dir, options)
    write_audit_markdown(report_dir / "data_cleaning_audit.md", audit_df, market, options)

    counts = audit_df["status"].value_counts().to_dict() if not audit_df.empty else {}
    print(f"market rows: {len(market)}")
    print(f"options rows: {len(options)}")
    print(f"data_cleaning_audit status counts: {counts}")
    print(f"wrote {out_dir / 'market.csv'}")
    print(f"wrote {out_dir / 'options.csv'}")
    print(f"wrote {report_dir / 'data_cleaning_audit.csv'}")
    print(f"wrote {report_dir / 'data_cleaning_audit.md'}")
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
        date_col = first_col(df, ["交易日期", "日期"])
        contract_col = first_col(df, ["契約"])
        expiry_col = first_col(df, ["到期月份(週別)", "到期月份"])
        if date_col is None or contract_col is None:
            continue
        tx = df[df[contract_col].astype(str).str.strip().eq("TX")].copy()
        if tx.empty:
            continue
        work = pd.DataFrame()
        work["date"] = parse_date(tx[date_col])
        work["contract"] = tx[contract_col].astype(str).str.strip()
        work["expiry_raw"] = tx[expiry_col].astype(str).str.strip() if expiry_col else ""
        work["expiry_key"] = expiry_sort_key(work["expiry_raw"])
        work["tx_open"] = to_number(col_or_nan(tx, ["開盤價"]))
        work["tx_high"] = to_number(col_or_nan(tx, ["最高價"]))
        work["tx_low"] = to_number(col_or_nan(tx, ["最低價"]))
        work["tx_close"] = to_number(col_or_nan(tx, ["收盤價", "最後成交價"]))
        work["txf_close"] = work["tx_close"]
        work["volume"] = to_number(col_or_nan(tx, ["成交量"])).fillna(0)
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


def load_official_options(opt_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(opt_dir.glob("*.csv")):
        df = read_csv_with_encoding(path)
        if df.empty:
            continue
        df = normalize_columns(df)
        date_col = first_col(df, ["交易日期", "日期"])
        if date_col is None:
            continue
        contract = col_or_nan(df, ["契約"]).astype(str).str.strip()
        txo = df[contract.eq("TXO")].copy()
        if txo.empty:
            continue
        out = pd.DataFrame()
        out["date"] = parse_date(txo[date_col])
        out["contract"] = col_or_nan(txo, ["契約"]).astype(str).str.strip()
        expiry_raw = col_or_nan(txo, ["到期月份(週別)", "到期月份"]).astype(str).str.strip()
        out["expiry_raw"] = expiry_raw
        expiry_lookup = {raw: parse_taifex_expiry(raw) for raw in expiry_raw.dropna().unique()}
        out["expiry"] = expiry_raw.map(expiry_lookup)
        out["dte"] = (pd.to_datetime(out["expiry"]) - pd.to_datetime(out["date"])).dt.days
        out["cp"] = col_or_nan(txo, ["買賣權"]).map(normalize_cp)
        out["strike"] = to_number(col_or_nan(txo, ["履約價"]))
        out["open"] = to_number(col_or_nan(txo, ["開盤價"]))
        out["high"] = to_number(col_or_nan(txo, ["最高價"]))
        out["low"] = to_number(col_or_nan(txo, ["最低價"]))
        out["close"] = to_number(col_or_nan(txo, ["最後成交價", "收盤價"]))
        out["bid"] = to_number(col_or_nan(txo, ["最後最佳買價"]))
        out["ask"] = to_number(col_or_nan(txo, ["最後最佳賣價"]))
        out["volume"] = to_number(col_or_nan(txo, ["成交量"])).fillna(0)
        out["open_interest"] = to_number(col_or_nan(txo, ["未沖銷契約量", "未沖銷契約數"]))
        out["settlement_price"] = to_number(col_or_nan(txo, ["結算價"]))
        out["halt_flag"] = col_or_nan(txo, ["是否因訊息面暫停交易"])
        out["session"] = col_or_nan(txo, ["交易時段"])
        out["is_weekly"] = expiry_raw.str.contains("W", case=False, na=False)
        out["iv"] = np.nan
        out["delta"] = np.nan
        out["iv_estimated"] = False
        out["delta_estimated"] = False
        out["bid_ask_estimated"] = False
        out["data_source"] = "taifex_official_daily"
        out["source_file"] = path.name
        frames.append(out[OPTION_COLUMNS + ["expiry_raw"]])
    return concat_or_empty(frames, OPTION_COLUMNS + ["expiry_raw"]).reset_index(drop=True)


def build_audit(market: pd.DataFrame, options: pd.DataFrame, txf_market: pd.DataFrame, official_market: pd.DataFrame) -> list[AuditItem]:
    rows: list[AuditItem] = []
    rows.extend(audit_market("market", market))
    rows.extend(audit_options("options", options))
    rows.append(AuditItem("market", "source_file_distribution", "PASS", distribution(market, "source_file")))
    rows.append(AuditItem("market", "data_source_distribution", "PASS", distribution(market, "data_source")))
    rows.append(AuditItem("market", "official_rows", "PASS" if len(official_market) > 0 else "WARN", f"rows={len(official_market)}"))
    rows.append(AuditItem("market", "crazyindicator_rows", "PASS" if len(txf_market) > 0 else "WARN", f"rows={len(txf_market)}"))
    rows.append(AuditItem("options", "weekly_monthly_distribution", "PASS", distribution(options, "is_weekly")))
    rows.append(AuditItem("options", "source_file_distribution", "PASS", distribution(options, "source_file")))
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
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except UnicodeDecodeError as exc:
            last_exc = exc
            continue
    if last_exc is not None:
        raise last_exc
    return pd.DataFrame()


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
    mapping = {"買權": "C", "賣權": "P", "CALL": "C", "PUT": "P", "C": "C", "P": "P"}
    return mapping.get(text, np.nan)


def expiry_sort_key(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.extract(r"(\d{6})", expand=False)
    return pd.to_numeric(text, errors="coerce").fillna(999999)


def parse_taifex_expiry(value: object) -> pd.Timestamp | pd.NaT:
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
