from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


def _prepare_real_data_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "prepare_real_data.py"
    spec = importlib.util.spec_from_file_location("prepare_real_data", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_taifex_expiry_supports_friday_weeklies() -> None:
    module = _prepare_real_data_module()

    assert module.parse_taifex_expiry("202507F3") == pd.Timestamp("2025-07-18")
    assert module.parse_taifex_expiry("202507F4") == pd.Timestamp("2025-07-25")


def test_parse_taifex_expiry_supports_wednesday_weeklies_and_monthlies() -> None:
    module = _prepare_real_data_module()

    assert module.parse_taifex_expiry("202501W1") == pd.Timestamp("2025-01-01")
    assert module.parse_taifex_expiry("202401") == pd.Timestamp("2024-01-17")


def test_parse_taifex_expiry_uses_override_when_available(tmp_path) -> None:
    module = _prepare_real_data_module()
    calendar_path = tmp_path / "txo_expiry_calendar.csv"
    calendar_path.write_text(
        "expiry_raw,actual_expiry,note\n202501W1,2025-01-02,New Year holiday adjustment\n",
        encoding="utf-8",
    )

    calendar = module.load_expiry_calendar(calendar_path)

    assert calendar.exists is True
    assert calendar.warnings == ()
    assert module.parse_taifex_expiry("202501W1", calendar.overrides) == pd.Timestamp("2025-01-02")


def test_parse_taifex_expiry_falls_back_when_override_missing(tmp_path) -> None:
    module = _prepare_real_data_module()
    calendar_path = tmp_path / "txo_expiry_calendar.csv"
    calendar_path.write_text(
        "expiry_raw,actual_expiry,note\n202507F3,2025-07-18,Friday weekly\n",
        encoding="utf-8",
    )

    calendar = module.load_expiry_calendar(calendar_path)

    assert module.parse_taifex_expiry("202501W1", calendar.overrides) == pd.Timestamp("2025-01-01")


def test_load_expiry_calendar_warns_on_invalid_rows(tmp_path) -> None:
    module = _prepare_real_data_module()
    calendar_path = tmp_path / "txo_expiry_calendar.csv"
    calendar_path.write_text(
        "expiry_raw,actual_expiry,note\nBAD,2025-01-02,bad raw\n202501W1,not-a-date,bad date\n202501W2,2024-12-31,before month\n202501W3,2025-01-15,valid\n",
        encoding="utf-8",
    )

    calendar = module.load_expiry_calendar(calendar_path)

    assert calendar.exists is True
    assert "202501W3" in calendar.overrides
    assert "BAD" not in calendar.overrides
    assert "202501W1" not in calendar.overrides
    assert "202501W2" not in calendar.overrides
    assert len(calendar.warnings) == 3


def test_classify_quote_quality_bid_gt_ask() -> None:
    module = _prepare_real_data_module()

    status, spread = module.classify_quote_quality(12, 10)

    assert status == "BID_GT_ASK"
    assert pd.isna(spread)


def test_classify_quote_quality_zero_bid_and_zero_ask() -> None:
    module = _prepare_real_data_module()

    assert module.classify_quote_quality(0, 10)[0] == "ZERO_BID"
    assert module.classify_quote_quality(10, 0)[0] == "ZERO_ASK"


def test_classify_quote_quality_extreme_spread_fail() -> None:
    module = _prepare_real_data_module()

    status, spread = module.classify_quote_quality(1, 4)

    assert status == "EXTREME_SPREAD_FAIL"
    assert spread > 1.0


def test_classify_quote_quality_extreme_spread_warn() -> None:
    module = _prepare_real_data_module()

    status, spread = module.classify_quote_quality(1, 2)

    assert status == "EXTREME_SPREAD_WARN"
    assert 0.5 < spread < 1.0


def test_classify_quote_quality_valid() -> None:
    module = _prepare_real_data_module()

    status, spread = module.classify_quote_quality(9, 10)

    assert status == "VALID"
    assert 0 <= spread <= 0.5


def test_is_tradable_quote_only_valid() -> None:
    module = _prepare_real_data_module()
    df = pd.DataFrame(
        {
            "bid": [9, 12, 0, 1, 1],
            "ask": [10, 10, 10, 4, 2],
        }
    )

    out = module.add_quote_quality_columns(df)

    assert out["is_tradable_quote"].tolist() == [True, False, False, False, False]
    assert out["quote_quality_status"].tolist() == [
        "VALID",
        "BID_GT_ASK",
        "ZERO_BID",
        "EXTREME_SPREAD_FAIL",
        "EXTREME_SPREAD_WARN",
    ]


def test_raw_options_inventory_handles_empty_raw_folder(tmp_path) -> None:
    module = _prepare_real_data_module()
    opt_dir = tmp_path / "opt"
    opt_dir.mkdir()
    processed = pd.DataFrame({"date": pd.to_datetime(["2019-01-02"])})

    inventory = module.build_raw_options_file_inventory(opt_dir, processed)
    coverage = inventory[inventory["metric"] == "raw_file_coverage"].iloc[0]

    assert int(coverage["raw_files_count"]) == 0
    assert coverage["coverage_status_2020"] == "RAW_DATA_MISSING"


def test_raw_options_inventory_detects_file_date_range(tmp_path) -> None:
    module = _prepare_real_data_module()
    opt_dir = tmp_path / "opt"
    opt_dir.mkdir()
    (opt_dir / "txo_2020.csv").write_text(
        "交易日期,契約,履約價,買賣權\n2020-01-02,TXO,10000,買權\n2020-12-31,TXO,10000,賣權\n",
        encoding="utf-8-sig",
    )

    inventory = module.build_raw_options_file_inventory(opt_dir, pd.DataFrame({"date": pd.to_datetime(["2020-01-02"])}))
    raw = inventory[inventory["section"] == "raw_file"].iloc[0]

    assert raw["date_min"] == "2020-01-02"
    assert raw["date_max"] == "2020-12-31"
    assert bool(raw["has_2020_rows"])


def test_raw_options_inventory_marks_raw_data_missing(tmp_path) -> None:
    module = _prepare_real_data_module()
    opt_dir = tmp_path / "opt"
    opt_dir.mkdir()
    (opt_dir / "txo_2019.csv").write_text(
        "交易日期,契約,履約價,買賣權\n2019-01-02,TXO,10000,買權\n",
        encoding="utf-8-sig",
    )

    inventory = module.build_raw_options_file_inventory(opt_dir, pd.DataFrame({"date": pd.to_datetime(["2019-01-02"])}))
    coverage = inventory[inventory["metric"] == "raw_file_coverage"].iloc[0]

    assert coverage["coverage_status_2020"] == "RAW_DATA_MISSING"


def test_raw_options_inventory_marks_ingestion_gap(tmp_path) -> None:
    module = _prepare_real_data_module()
    opt_dir = tmp_path / "opt"
    opt_dir.mkdir()
    (opt_dir / "txo_2020.csv").write_text(
        "交易日期,契約,履約價,買賣權\n2020-01-02,TXO,10000,買權\n",
        encoding="utf-8-sig",
    )
    processed = pd.DataFrame({"date": pd.to_datetime(["2019-01-02"])})

    inventory = module.build_raw_options_file_inventory(opt_dir, processed)
    coverage = inventory[inventory["metric"] == "raw_file_coverage"].iloc[0]

    assert coverage["coverage_status_2020"] == "INGESTION_GAP"


def test_raw_options_inventory_report_has_no_best_or_recommend(tmp_path) -> None:
    module = _prepare_real_data_module()
    path = tmp_path / "dummy.md"
    text_frame = pd.DataFrame([{"section": "aggregate_summary", "metric": "raw_file_coverage", "coverage_status_2020": "RAW_DATA_MISSING"}])
    module.write_raw_options_inventory_markdown(path, text_frame)
    text = path.read_text(encoding="utf-8").lower()

    assert "best" not in text
    assert "recommend" not in text
