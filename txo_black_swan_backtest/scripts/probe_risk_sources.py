from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.risk_source_probe import probe_risk_sources  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe registered official risk indicator sources.")
    parser.add_argument("--out-dir", default="reports")
    parser.add_argument("--registry", default="data/risk_sources.yml")
    parser.add_argument("--insecure", action="store_true", help="Retry HTTPS probes without certificate verification.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    probe = probe_risk_sources(ROOT / args.registry, ROOT / args.out_dir, insecure=args.insecure)
    print(f"risk_source_probe rows={len(probe)}")
    print(f"output={ROOT / args.out_dir / 'risk_source_probe.csv'}")


if __name__ == "__main__":
    main()
