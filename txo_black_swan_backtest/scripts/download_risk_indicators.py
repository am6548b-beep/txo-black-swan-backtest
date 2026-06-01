from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.risk_indicator_downloader import download_risk_indicators  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download official risk indicators into processed CSV format.")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--out-dir", default="data/raw/risk_indicators")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--report-dir", default="reports")
    parser.add_argument("--registry", default="data/risk_sources.yml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    risk, audit = download_risk_indicators(
        args.start_date,
        args.end_date,
        ROOT / args.out_dir,
        ROOT / args.processed_dir,
        ROOT / args.report_dir,
        ROOT / args.registry,
    )
    failed = audit[audit["status"].astype(str) == "failed"]["source_name"].tolist() if not audit.empty else []
    print(f"risk_indicators.csv rows={len(risk)}")
    print(f"audit rows={len(audit)}")
    print(f"failed sources={failed}")


if __name__ == "__main__":
    main()
