#!/usr/bin/env python3
"""E008b VNM metric-row grounding fixture."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifinqa.retrieval import load_reports
from vifinqa.row_retrieval_v2 import retrieve_rows_clean


def main() -> None:
    reports = load_reports(ROOT / "data" / "raw" / "vifinqa")
    result = retrieve_rows_clean(
        "Doanh thu thuần của VNM năm 2023 là bao nhiêu VND?",
        {"tickers": ["VNM"], "years": [2023], "scope": "consolidated"}, reports, top_k=10,
    )
    matches = [row for row in result["tables"] if row["table_id"].endswith("|237") and "Doanh thu thuần" in " ".join(row["row_cells"])]
    assert matches, result["tables"]
    print("E008b fixture passed")


if __name__ == "__main__":
    main()

