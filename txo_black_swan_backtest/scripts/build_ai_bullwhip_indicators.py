from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ai_bullwhip_data_loader import build_ai_bullwhip_indicators  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build manual AI bullwhip indicators.")
    parser.add_argument("--raw-dir", default="data/raw/ai_bullwhip")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--report-dir", default="reports")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed, audit = build_ai_bullwhip_indicators(ROOT / args.raw_dir, ROOT / args.processed_dir, ROOT / args.report_dir)
    print(f"ai_bullwhip_indicators.csv rows={len(processed)}")
    if not processed.empty:
        print(f"date range={processed['date'].min()} to {processed['date'].max()}")
    print(f"audit rows={len(audit)}")
    print(f"output={ROOT / args.processed_dir / 'ai_bullwhip_indicators.csv'}")


if __name__ == "__main__":
    main()
