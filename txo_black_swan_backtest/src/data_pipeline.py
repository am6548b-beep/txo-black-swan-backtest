"""Raw-to-processed data pipeline for real TAIFEX and macro CSV files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .data_loader import MACRO_FACTOR_COLUMNS, OPTION_COLUMNS
from .taifex_loader import build_market_frame, load_macro_factors_raw, load_taifex_futures, load_taifex_options, load_taifex_vix


def build_processed_data(raw_dir: Path, processed_dir: Path) -> dict[str, int | str]:
    """Convert raw CSV inputs into normalized processed CSV files."""

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

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
        "raw_dir": str(raw_dir),
        "processed_dir": str(processed_dir),
        "market_rows": int(len(market)),
        "options_rows": int(len(options)),
        "macro_rows": int(len(macro)),
    }
