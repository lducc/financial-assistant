#!/usr/bin/env python3
"""Create one submission-ready evidence CSV from a line-addressed OCR table."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifinqa.evidence import materialize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--start-line", type=int, required=True)
    parser.add_argument("--table-id", required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    args = parser.parse_args()
    table = materialize(args.report, args.start_line, args.table_id, args.csv, args.sidecar)
    print(f"Materialized {table.table_id}: {len(table.rows)} rows")


if __name__ == "__main__":
    main()

