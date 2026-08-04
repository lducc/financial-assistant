#!/usr/bin/env python3
"""E010 validator acceptance tests."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifinqa.annotation import validate_record


def complete() -> dict:
    return {"id": 1, "annotation": {
        "status": "complete", "annotator": "reviewer_a", "question_slots": {"metric": "revenue"},
        "required_metric_roles": ["value"], "gold_reports": ["VNM_financial_statements_2023_consolidated"],
        "gold_tables": ["VNM_financial_statements_2023_consolidated|237"],
        "row_column_bindings": [{"row": "Revenue", "column": "2023"}], "table_units": "million VND",
        "operation_graph": {"op": "lookup"}, "pandas_query": "result = float(df.iloc[0, 1])",
        "numeric_answer": 123.4, "failure_tags": [], "confidence": "high",
    }}


def main() -> None:
    blank = {"id": 2, "annotation": {"status": "unannotated"}}
    assert not validate_record(blank, allow_incomplete=True)
    assert validate_record(blank, allow_incomplete=False)
    assert not validate_record(complete())
    bad = complete()
    bad["annotation"]["gold_tables"] = ["OTHER|22"]
    assert any("missing from gold_reports" in error for error in validate_record(bad))
    bad = complete()
    bad["annotation"]["pandas_query"] = "42"
    assert any("assign result" in error for error in validate_record(bad))
    print("E010 annotation validator tests passed")


if __name__ == "__main__":
    main()
