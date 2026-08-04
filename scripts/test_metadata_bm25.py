#!/usr/bin/env python3
"""E007 integration check on a known VNM revenue question."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifinqa.retrieval import load_reports, retrieve


def main() -> None:
    reports = load_reports(ROOT / "data" / "raw" / "vifinqa")
    result = retrieve(
        "Doanh thu thuần của VNM năm 2023 là bao nhiêu VND?",
        {"tickers": ["VNM"], "years": [2023], "scope": "consolidated"},
        reports,
        top_k=5,
    )
    assert result["filter_stage"] == "ticker_year_scope"
    assert result["candidate_report_count"] == 1
    assert any(table["table_id"].endswith("|237") for table in result["tables"])
    print("E007 fixture passed")


if __name__ == "__main__":
    main()

