#!/usr/bin/env python3
"""E013 agreement acceptance tests."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifinqa.agreement import compare_reviews


def record(identifier: int, answer: float = 100.0, table: str = "VNM_report|237", unit: str = "million VND") -> dict:
    return {"id": identifier, "annotation": {"gold_reports": ["VNM_report"], "gold_tables": [table], "operation_graph": {"op": "lookup"}, "table_units": unit, "numeric_answer": answer}}


def main() -> None:
    identical = compare_reviews([record(1)], [record(1)])
    assert identical["agreement"] == {"report_set_exact": 1, "table_set_exact": 1, "operation_graph_exact": 1, "unit_exact": 1, "numeric_within_0_02_percent": 1}
    inside = compare_reviews([record(1, 100)], [record(1, 100.019)])
    assert inside["pairs"][0]["numeric_within_0_02_percent"]
    outside = compare_reviews([record(1, 100)], [record(1, 100.03, "VNM_report|238", "billion VND"), record(2)])
    assert not outside["pairs"][0]["numeric_within_0_02_percent"]
    assert not outside["pairs"][0]["table_set_exact"] and not outside["pairs"][0]["unit_exact"]
    assert outside["right_only_ids"] == [2]
    print("E013 annotation agreement tests passed")


if __name__ == "__main__":
    main()
