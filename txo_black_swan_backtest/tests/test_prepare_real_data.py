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


def test_recognition_debug_recurses_and_parses_roc_dates(tmp_path) -> None:
    module = _prepare_real_data_module()
    opt_dir = tmp_path / "opt"
    nested = opt_dir / "nested"
    nested.mkdir(parents=True)
    (nested / "roc_2020.txt").write_text(
        "交易日期,契約,履約價,買賣權\n109/01/02,TXO,10000,買權\n109/03/02,TXO,10000,賣權\n",
        encoding="utf-8-sig",
    )

    debug = module.build_raw_options_file_recognition_debug(opt_dir, pd.DataFrame({"date": pd.to_datetime(["2019-01-02"])}))
    raw = debug[debug["section"] == "raw_file_recognition"].iloc[0]

    assert raw["parsed_date_min"] == "2020-01-02"
    assert bool(raw["roc_year_detected"])
    assert bool(raw["has_2020_rows"])


def test_recognition_debug_marks_ingestion_recognition_gap(tmp_path) -> None:
    module = _prepare_real_data_module()
    opt_dir = tmp_path / "opt"
    opt_dir.mkdir()
    (opt_dir / "western_2020.csv").write_text(
        "交易日期,契約,履約價,買賣權\n2020-01-02,TXO,10000,買權\n",
        encoding="utf-8-sig",
    )

    debug = module.build_raw_options_file_recognition_debug(opt_dir, pd.DataFrame({"date": pd.to_datetime(["2019-01-02"])}))
    status = debug[(debug["section"] == "aggregate_summary") & (debug["metric"] == "year_coverage_status") & (debug["year"] == 2020)].iloc[0]

    assert status["coverage_status"] == "INGESTION_RECOGNITION_GAP"


def test_recognition_debug_marks_raw_data_missing(tmp_path) -> None:
    module = _prepare_real_data_module()
    opt_dir = tmp_path / "opt"
    opt_dir.mkdir()

    debug = module.build_raw_options_file_recognition_debug(opt_dir, pd.DataFrame({"date": pd.to_datetime(["2019-01-02"])}))
    status = debug[(debug["section"] == "aggregate_summary") & (debug["metric"] == "year_coverage_status") & (debug["year"] == 2020)].iloc[0]

    assert status["coverage_status"] == "RAW_DATA_MISSING"


def test_recognition_debug_report_has_no_best_or_recommend(tmp_path) -> None:
    module = _prepare_real_data_module()
    path = tmp_path / "recognition.md"
    frame = pd.DataFrame([{"section": "aggregate_summary", "metric": "raw_files_count", "value": 0}])

    module.write_raw_options_recognition_debug_markdown(path, frame)
    text = path.read_text(encoding="utf-8").lower()

    assert "best" not in text
    assert "recommend" not in text


def test_new_schema_option_columns_normalize(tmp_path) -> None:
    module = _prepare_real_data_module()
    opt_dir = tmp_path / "opt"
    opt_dir.mkdir()
    (opt_dir / "new_schema.csv").write_text(
        "交易日期,契約,到期月份(週別),履約價,買賣權,開盤價,最高價,最低價,收盤價,成交量,結算價,未沖銷契約數,最後最佳買價,最後最佳賣價,是否因訊息面暫停交易,交易時段\n"
        "2020/01/02,CAO,202003,10000,買權,10,12,8,11,100,11,200,10,12,,一般\n"
        "2020/01/02,CAO,202003,10000,賣權,9,11,7,10,100,10,200,9,11,,一般\n",
        encoding="utf-8-sig",
    )

    out = module.load_official_options(opt_dir)

    assert len(out) == 2
    assert set(out["cp"]) == {"C", "P"}
    assert set(["date", "expiry", "strike", "cp", "bid", "ask"]) <= set(out.columns)


def test_english_schema_option_columns_normalize(tmp_path) -> None:
    module = _prepare_real_data_module()
    opt_dir = tmp_path / "opt"
    opt_dir.mkdir()
    (opt_dir / "english_schema.csv").write_text(
        "Date,Contract,Expiry,Strike,CP,Open,High,Low,Close,Volume,Settlement,OI,Bid,Ask,Session\n"
        "2020-01-02,TXO,202003,10000,C,10,12,8,11,100,11,200,10,12,regular\n",
        encoding="utf-8-sig",
    )

    out = module.load_official_options(opt_dir)

    assert len(out) == 1
    assert out.iloc[0]["cp"] == "C"
    assert out.iloc[0]["bid"] == 10
    assert out.iloc[0]["ask"] == 12


def test_schema_alias_does_not_override_existing_correct_column() -> None:
    module = _prepare_real_data_module()
    df = pd.DataFrame({"Date": ["2020-01-02"], "TradingDate": ["1999-01-01"], "Contract": ["TXO"]})

    schema = module.detect_option_schema(df)

    assert schema["date"] == "Date"


def test_2018_2024_mock_schema_outputs_core_fields(tmp_path) -> None:
    module = _prepare_real_data_module()
    opt_dir = tmp_path / "opt"
    opt_dir.mkdir()
    (opt_dir / "mock_2024.csv").write_text(
        "交易日期,契約,到期月份(週別),履約價,買賣權,收盤價,成交量,未沖銷契約數,最後最佳買價,最後最佳賣價\n"
        "2024/01/02,CAO,202403,18000,買權,100,10,100,99,101\n",
        encoding="utf-8-sig",
    )

    out = module.load_official_options(opt_dir)

    assert pd.notna(out.iloc[0]["date"])
    assert pd.notna(out.iloc[0]["expiry"])
    assert out.iloc[0]["strike"] == 18000
    assert out.iloc[0]["cp"] == "C"
    assert out.iloc[0]["bid"] == 99
    assert out.iloc[0]["ask"] == 101


def test_schema_mapping_audit_report_has_no_best_or_recommend(tmp_path) -> None:
    module = _prepare_real_data_module()
    path = tmp_path / "schema.md"
    audit = pd.DataFrame([{"status": "PASS", "source_file": "x.csv", "missing_required_columns": "", "issue": ""}])

    module.write_schema_mapping_audit_markdown(path, audit)
    text = path.read_text(encoding="utf-8").lower()

    assert "best" not in text
    assert "recommend" not in text
