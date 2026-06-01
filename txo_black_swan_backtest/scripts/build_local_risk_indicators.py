from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.risk_indicator_builder import build_local_risk_indicators  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build local TXO-chain derived risk indicators.")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--report-dir", default="reports")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    indicators, audit = build_local_risk_indicators(ROOT / args.processed_dir, ROOT / args.report_dir)
    print(f"risk_indicators.csv rows={len(indicators)}")
    print(f"audit rows={len(audit)}")
    print(f"output={ROOT / args.processed_dir / 'risk_indicators.csv'}")


if __name__ == "__main__":
    main()
