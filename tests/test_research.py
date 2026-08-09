"""Focused tests for OCR parsing, retrieval, and pilot source bindings."""

import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifinqa.retrieval import Report, report_tables, retrieve_rows
from vifinqa.review import validate_source_bindings
from vifinqa.tables import ReportIdentity, parse_table_rows


def test_html_spans_expand_grid():
    html = '<table><tr><th rowspan="2">Metric</th><th colspan="2">Balance</th></tr><tr><td>2024</td><td>2023</td></tr></table>'
    assert parse_table_rows(html) == [["Metric", "Balance", "Balance"], ["Metric", "2024", "2023"]]


def test_retrieval_produces_unique_table_ids(tmp_path):
    path = tmp_path / "report.txt"
    path.write_text(
        "Bảng doanh thu\n<table><tr><td>Chỉ tiêu</td><td>2024</td></tr><tr><td>Doanh thu</td><td>42</td></tr></table>\n"
        "Bảng tài sản\n<table><tr><td>Chỉ tiêu</td><td>2024</td></tr><tr><td>Tài sản</td><td>1</td></tr></table>",
        encoding="utf-8",
    )
    identity = ReportIdentity("R", "ABC", 2024, "consolidated", path.name)
    result = retrieve_rows("Doanh thu năm 2024", {"years": [2024]}, [Report(identity, path)], top_k=5)
    assert [table["table_id"] for table in result["tables"]] == ["R|2", "R|4"]
    assert len({table["table_id"] for table in result["tables"]}) == len(result["tables"])


def test_source_binding_validation_reads_raw_table(tmp_path):
    report_id = "TEST_financial_statements_2024_consolidated"
    source = tmp_path / "financial_statements" / "TEST" / "2024" / report_id / f"{report_id}_extracted.txt"
    source.parent.mkdir(parents=True)
    source.write_text("heading\n<table><tr><td>Metric</td><td>Value</td></tr><tr><td>Revenue</td><td>42</td></tr></table>", encoding="utf-8")
    record = {"annotation": {"status": "complete", "row_column_bindings": [{
        "table": f"{report_id}|2", "row": 1, "column": 1, "raw": "42",
    }]}}
    assert validate_source_bindings(record, tmp_path) == []
    record["annotation"]["row_column_bindings"][0]["raw"] = "wrong"
    assert "raw cell mismatch" in validate_source_bindings(record, tmp_path)[0]


def test_retained_pilot_has_valid_schema_and_no_gold_reference():
    records = [json.loads(line) for line in (ROOT / "annotations" / "pilot_v1" / "agent_labels.jsonl").read_text(encoding="utf-8").splitlines() if line]
    assert len(records) == 12
    assert all(record["review"]["independent_from_gold_150"] for record in records)


def test_evaluator_fixed_prefix_metrics():
    spec = importlib.util.spec_from_file_location("evaluator", ROOT / "scripts" / "evaluate_table_retrieval.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    assert module.prefix_score(["A", "B"], ["A", "X", "B"], 1)["recall"] == 0.5
