"""Raw-to-processed data pipeline for real TAIFEX and macro CSV files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .data_loader import MACRO_FACTOR_COLUMNS, OPTION_COLUMNS
from .taifex_loader import build_market_frame, load_macro_factors_raw, load_taifex_futures, load_taifex_options, load_taifex_vix


def build_processed_data(raw_dir: Path, processed_dir: Path, *, overwrite: bool = False, sample_dir: Path | None = None) -> dict[str, bool | int | str]:
    """Convert raw CSV inputs into normalized processed CSV files."""

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    raw_files = [p for p in raw_dir.rglob("*.csv") if p.is_file()]
    has_existing_processed = _has_existing_processed(processed_dir)
    sample_status = "USING_SAMPLE_DATA" if sample_dir is not None and _has_existing_processed(sample_dir) else "NO_SAMPLE_DATA"
    if not raw_files:
        return {
            "status": "NO_RAW_DATA",
            "data_source_status": "USING_EXISTING_PROCESSED_DATA" if has_existing_processed else sample_status,
            "raw_dir": str(raw_dir),
            "processed_dir": str(processed_dir),
            "market_rows": _existing_rows(processed_dir / "market.csv"),
            "options_rows": _existing_rows(processed_dir / "options.csv"),
            "macro_rows": _existing_rows(processed_dir / "macro_factors.csv"),
            "overwritten": False,
            "real_data_backtest": False,
            "note": "This is not a real-data backtest. raw directory has no csv files; existing processed data was not overwritten.",
        }

    if has_existing_processed and not overwrite:
        return {
            "status": "FOUND_RAW_DATA",
            "data_source_status": "USING_EXISTING_PROCESSED_DATA",
            "raw_dir": str(raw_dir),
            "processed_dir": str(processed_dir),
            "raw_csv_files": int(len(raw_files)),
            "market_rows": _existing_rows(processed_dir / "market.csv"),
            "options_rows": _existing_rows(processed_dir / "options.csv"),
            "macro_rows": _existing_rows(processed_dir / "macro_factors.csv"),
            "overwritten": False,
            "real_data_backtest": False,
            "note": "Raw CSV files were found, but existing processed data was not overwritten. Rerun with --overwrite or config allow_processed_overwrite=True to rebuild processed data.",
        }

    futures = load_taifex_futures(raw_dir)
    vix = load_taifex_vix(raw_dir)
    options = load_taifex_options(raw_dir)
    macro = load_macro_factors_raw(raw_dir)
    market = build_market_frame(futures, vix)

    if macro.empty:
        macro = pd.DataFrame(columns=MACRO_FACTOR_COLUMNS)
    if options.empty:
        options = pd.DataFrame(columns=OPTION_COLUMNS)

    market.to_csv(processed_dir / "market.csv", index=False)
    options.to_csv(processed_dir / "options.csv", index=False)
    macro.to_csv(processed_dir / "macro_factors.csv", index=False)

    return {
        "status": "FOUND_RAW_DATA",
        "data_source_status": "FOUND_RAW_DATA",
        "raw_dir": str(raw_dir),
        "processed_dir": str(processed_dir),
        "raw_csv_files": int(len(raw_files)),
        "market_rows": int(len(market)),
        "options_rows": int(len(options)),
        "macro_rows": int(len(macro)),
        "overwritten": bool(has_existing_processed),
        "real_data_backtest": True,
        "note": "Processed data was rebuilt from raw CSV files.",
    }


def _existing_rows(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    try:
        return int(len(pd.read_csv(path)))
    except Exception:
        return 0


def _has_existing_processed(processed_dir: Path) -> bool:
    return (processed_dir / "market.csv").exists() and (processed_dir / "options.csv").exists()
