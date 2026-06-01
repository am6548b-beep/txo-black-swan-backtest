"""Command-line entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from config import BASE_CONFIG, COARSE_PARAM_GRID, IRON_CONDOR_PARAMS, PUT_SPREAD_PARAMS, REGIME_WINDOWS, WALK_FORWARD_WINDOWS
from src.backtester import run_backtest
from src.data_pipeline import build_processed_data
from src.data_validation import validate_data
from src.hedge_need import write_hedge_need_diagnostics
from src.pathing import ensure_data_layout, resolve_processed_data_dir, resolve_raw_data_dir, resolve_runtime_data_dir
from src.put_spread_analysis import write_put_spread_real_data_analysis
from src.put_spread_coverage import write_put_spread_coverage_audit
from src.put_spread_variants import run_put_spread_variants
from src.reports import write_reports
from src.report_auditor import run_report_audit
from src.stress_tests import run_stress_suite
from src.utils import ensure_dirs, load_config_module
from src.walk_forward import run_walk_forward


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TXO black-swan hedge and post-panic short-vol backtester")
    parser.add_argument("--config", default="config.py", help="Path to config.py")
    parser.add_argument(
        "--mode",
        default="full",
        choices=[
            "put_spread_only",
            "iron_condor_only",
            "full",
            "stress",
            "walk_forward",
            "validate_data",
            "build_data",
            "build_dataset",
            "report_audit",
            "put_spread_analysis",
            "put_spread_coverage",
            "put_spread_variants",
            "hedge_need_diagnostics",
        ],
    )
    parser.add_argument("--data-dir", default=None, help="Override runtime data directory")
    parser.add_argument("--raw-dir", default=None, help="Override raw input directory for build_data")
    parser.add_argument("--processed-dir", default=None, help="Override processed output directory")
    parser.add_argument("--overwrite", action="store_true", help="Allow build_dataset to overwrite existing processed CSV files")
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

    if args.mode in {"build_data", "build_dataset"}:
        overwrite = bool(args.overwrite or config.get("allow_processed_overwrite", False))
        result = build_processed_data(raw_data_dir, processed_data_dir, overwrite=overwrite, sample_dir=root / "data" / "sample")
        print(f"Processed data dir: {processed_data_dir}")
        print(result)
        return

    if args.mode == "report_audit":
        audit = run_report_audit(root / "reports")
        write_put_spread_real_data_analysis(root / "reports", processed_data_dir)
        write_put_spread_coverage_audit(root / "reports", processed_data_dir, config, put_params)
        counts = audit["status"].value_counts().to_dict() if not audit.empty else {}
        print(f"Report audit written to {root / 'reports' / 'report_audit.csv'}")
        print(f"Report audit markdown written to {root / 'reports' / 'report_audit.md'}")
        print(f"Put spread analysis written to {root / 'reports' / 'put_spread_real_data_analysis.csv'}")
        print(f"Put spread coverage audit written to {root / 'reports' / 'put_spread_coverage_audit.csv'}")
        print(f"Status counts: {counts}")
        return

    if args.mode == "put_spread_analysis":
        analysis = write_put_spread_real_data_analysis(root / "reports", processed_data_dir)
        print(f"Put spread analysis written to {root / 'reports' / 'put_spread_real_data_analysis.csv'}")
        print(f"Rows: {len(analysis)}")
        return

    if args.mode == "put_spread_coverage":
        coverage, entry = write_put_spread_coverage_audit(root / "reports", processed_data_dir, config, put_params)
        print(f"Put spread coverage audit written to {root / 'reports' / 'put_spread_coverage_audit.csv'}")
        print(f"Put spread entry signal audit written to {root / 'reports' / 'put_spread_entry_signal_audit.csv'}")
        print(f"Rows: coverage={len(coverage)}, entry={len(entry)}")
        return

    if args.mode == "put_spread_variants":
        comparison = run_put_spread_variants(processed_data_dir, root / "reports", config, put_params, ic_params)
        print(f"Put spread variant comparison written to {root / 'reports' / 'put_spread_variant_comparison.csv'}")
        print(f"Rows: {len(comparison)}")
        return

    if args.mode == "hedge_need_diagnostics":
        score, coverage, audit = write_hedge_need_diagnostics(processed_data_dir, root / "reports", config, put_params)
        print(f"Hedge need score written to {root / 'reports' / 'hedge_need_score.csv'}")
        print(f"Hedge coverage timeline written to {root / 'reports' / 'hedge_coverage_timeline.csv'}")
        print(f"Hedge gap audit written to {root / 'reports' / 'hedge_gap_audit.csv'}")
        print(f"Rows: score={len(score)}, coverage={len(coverage)}, audit={len(audit)}")
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
