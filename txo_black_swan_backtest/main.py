"""Command-line entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from config import BASE_CONFIG, COARSE_PARAM_GRID, IRON_CONDOR_PARAMS, PUT_SPREAD_PARAMS, REGIME_WINDOWS, WALK_FORWARD_WINDOWS
from src.backtester import run_backtest
from src.data_pipeline import build_processed_data
from src.data_validation import validate_data
from src.pathing import ensure_data_layout, resolve_processed_data_dir, resolve_raw_data_dir, resolve_runtime_data_dir
from src.reports import write_reports
from src.stress_tests import run_stress_suite
from src.utils import ensure_dirs, load_config_module
from src.walk_forward import run_walk_forward


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TXO black-swan hedge and post-panic short-vol backtester")
    parser.add_argument("--config", default="config.py", help="Path to config.py")
    parser.add_argument("--mode", default="full", choices=["put_spread_only", "iron_condor_only", "full", "stress", "walk_forward", "validate_data", "build_data"])
    parser.add_argument("--data-dir", default=None, help="Override runtime data directory")
    parser.add_argument("--raw-dir", default=None, help="Override raw input directory for build_data")
    parser.add_argument("--processed-dir", default=None, help="Override processed output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    ensure_dirs(root)
    ensure_data_layout(root)

    cfg_module = load_config_module(root / args.config if not Path(args.config).is_absolute() else args.config)
    config = getattr(cfg_module, "BASE_CONFIG", BASE_CONFIG).copy()
    if args.data_dir:
        config["data_dir"] = args.data_dir
    if args.raw_dir:
        config["raw_data_dir"] = args.raw_dir
    if args.processed_dir:
        config["processed_data_dir"] = args.processed_dir
    put_params = getattr(cfg_module, "PUT_SPREAD_PARAMS", PUT_SPREAD_PARAMS).copy()
    ic_params = getattr(cfg_module, "IRON_CONDOR_PARAMS", IRON_CONDOR_PARAMS).copy()
    regimes = getattr(cfg_module, "REGIME_WINDOWS", REGIME_WINDOWS)
    runtime_data_dir = resolve_runtime_data_dir(root, config)
    processed_data_dir = resolve_processed_data_dir(root, config)
    raw_data_dir = resolve_raw_data_dir(root, config)

    if args.mode == "validate_data":
        report = validate_data(processed_data_dir, root / "reports", config)
        status_counts = report["status"].value_counts().to_dict() if not report.empty else {}
        print(f"Validated processed data dir: {processed_data_dir}")
        print(f"Data quality report written to {root / 'reports' / 'data_quality_report.csv'}")
        print(f"Status counts: {status_counts}")
        return

    if args.mode == "build_data":
        result = build_processed_data(raw_data_dir, processed_data_dir)
        print(f"Processed data written to {processed_data_dir}")
        print(result)
        return

    if args.mode == "stress":
        stress = run_stress_suite(config, put_params, ic_params)
        write_reports(root / "reports", pd.DataFrame(), pd.DataFrame(), regimes, label="stress", stress=stress)
        print(f"Stress report written to {root / 'reports'}")
        return

    if args.mode == "walk_forward":
        summary, overfit = run_walk_forward(
            runtime_data_dir,
            config,
            put_params,
            ic_params,
            getattr(cfg_module, "COARSE_PARAM_GRID", COARSE_PARAM_GRID),
            getattr(cfg_module, "WALK_FORWARD_WINDOWS", WALK_FORWARD_WINDOWS),
        )
        (root / "reports").mkdir(exist_ok=True)
        summary.to_csv(root / "reports" / "summary.csv", index=False)
        overfit.to_csv(root / "reports" / "overfit_risk_report.csv", index=False)
        print(f"Walk-forward reports written to {root / 'reports'}")
        return

    print(f"Using data dir: {runtime_data_dir}")
    equity, trades, market = run_backtest(runtime_data_dir, config, put_params, ic_params, mode=args.mode)
    write_reports(root / "reports", equity, trades, regimes, label=args.mode, market=market)
    print(f"Reports written to {root / 'reports'}")


if __name__ == "__main__":
    main()
