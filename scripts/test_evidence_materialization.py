#!/usr/bin/env python3
"""E006c integration fixture against a known literal VNM statement table."""

import csv
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifinqa.evidence import materialize


def main() -> None:
    report = ROOT / "data" / "raw" / "vifinqa" / "financial_statements" / "VNM" / "2023" / "VNM_financial_statements_2023_consolidated" / "VNM_financial_statements_2023_consolidated_extracted.txt"
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        table = materialize(report, 237, "VNM_financial_statements_2023_consolidated|237", directory / "evidence.csv", directory / "evidence.json")
        assert table.rows[0][1] == "Mã số"
        assert any("Doanh thu thuần" in cell for row in table.rows for cell in row)
        with (directory / "evidence.csv").open(encoding="utf-8") as handle:
            assert len(list(csv.reader(handle))) == len(table.rows)
        assert (directory / "evidence.json").is_file()
    print("E006c fixture passed")


if __name__ == "__main__":
    main()

