"""Focused tests for OCR parsing, retrieval, and pilot source bindings."""

import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifinqa.retrieval import Report, metric_query_tokens, rank_fuse_signal_scores, report_tables, retrieve_rows, table_budget
from vifinqa.answers import EvidenceValue, answer_plan, first_numeric_cell, parse_ocr_number
from vifinqa.rerank import table_representation
from vifinqa.dense import fused_rankings
import vifinqa.retrieval as retrieval_module
from vifinqa.evaluation_v2 import (
    atomic_write_run, build_frame, evaluate_v2_predictions, manifest, normalize_template,
    paired_cluster_bootstrap, sample_splits, score_slots, summarize_v2_traces,
    promotion_gate, validate_adjudication, validate_annotation_batch, validate_v2_annotation,
)
from vifinqa.review import validate_source_bindings
from vifinqa.tables import ReportIdentity, parse_table_rows


def test_html_spans_expand_grid():
    html = '<table><tr><th rowspan="2">Metric</th><th colspan="2">Balance</th></tr><tr><td>2024</td><td>2023</td></tr></table>'
    assert parse_table_rows(html) == [["Metric", "Balance", "Balance"], ["Metric", "2024", "2023"]]


def test_ocr_number_parser_handles_grouping_decimals_and_parentheses():
    assert parse_ocr_number("1.234.567") == 1234567.0
    assert parse_ocr_number("(1.234,5)") == -1234.5
    assert first_numeric_cell(["Metric", "-", "42,5%"]) == (2, 42.5)


def test_first_numeric_cell_skips_account_code_and_note_columns():
    assert first_numeric_cell(["Doanh thu thuan", "01", "VI.25", "1.234.567"]) == (3, 1234567.0)
    cells = ["Tien", "110", "V.01", "38.446.527.451"]
    assert first_numeric_cell(cells, ["Chi tieu", "Ma so", "Thuyet minh", "31/12/2020"]) == (3, 38446527451.0)
    assert first_numeric_cell(cells, ["Chỉ tiêu", "Mã số", "Thuyết minh", "31/12/2020"]) == (3, 38446527451.0)
    assert first_numeric_cell(["Doanh thu", "42"]) == (1, 42.0)
    assert first_numeric_cell(["Doanh thu", "-", "n/a"]) is None


def test_metric_query_tokens_remove_arithmetic_boilerplate_but_keep_line_item():
    tokens = metric_query_tokens(
        "Tính tỷ lệ tăng trưởng doanh thu thuần năm 2024 so với năm 2023 là bao nhiêu?",
        {"years": [2024, 2023], "tickers": []},
    )
    assert tokens == ["doanh", "thu", "thuan"]


def test_answer_plan_uses_only_bound_evidence_for_common_operations():
    values = [EvidenceValue("df0", 1, 2, 10.0, "R1"), EvidenceValue("df1", 2, 3, 30.0, "R2")]
    answer, expression = answer_plan("Tổng giá trị là bao nhiêu?", values) or (None, "")
    assert answer == 40.0 and "df0" in expression and "df1" in expression
    answer, _ = answer_plan("Tỷ lệ phần trăm là bao nhiêu?", values) or (None, "")
    assert answer is not None and abs(answer - 100 / 3) < 1e-9


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
    assert [
        (table["score"], table["row_index"], table["row_cells"])
        for table in result["tables"]
    ] == [
        (0.032787, 1, ["Doanh thu", "42"]),
        (0.016129, 0, ["Chỉ tiêu", "2024"]),
    ]
    assert len({table["table_id"] for table in result["tables"]}) == len(result["tables"])


def test_rank_fusion_is_scale_invariant_and_ignores_zero_signals():
    signals = {
        "folded_row": {"A": 9.0, "B": 3.0},
        "unicode_row": {"A": 4.0, "B": 2.0},
        "folded_context": {"B": 7.0, "A": 1.0},
        "unicode_context": {"B": 5.0, "A": 1.0},
        "title": {"A": 8.0, "B": 1.0},
        "header": {"A": 6.0, "B": 1.0},
        "unit": {"A": 0.0, "B": 0.0},
    }
    baseline = rank_fuse_signal_scores(signals)
    scaled = rank_fuse_signal_scores({
        **signals,
        "title": {table_id: score * 1_000_000 for table_id, score in signals["title"].items()},
        "unit": {"A": 0.0, "B": 0.0, "C": 0.0},
    })
    assert baseline == scaled
    assert "unit" not in baseline[1]["A"]
    assert "C" not in baseline[0]


def test_dense_fusion_unions_candidates_and_keeps_sparse_row_binding():
    class Candidate:
        def __init__(self, table_id):
            self.table_id = table_id

    alpha, beta, gamma = Candidate("A"), Candidate("B"), Candidate("C")
    fused = fused_rankings(
        [(9.0, alpha, 4), (8.0, beta, 2)],
        [(0.9, beta, 7), (0.8, alpha, 1), (0.7, gamma, 3)],
    )
    by_id = {table.table_id: row_index for _, table, row_index in fused}
    assert set(by_id) == {"A", "B", "C"}
    assert by_id["A"] == 4


def test_rank_fusion_ties_are_deterministic_and_families_are_balanced():
    tie_totals, tie_ranks, _ = rank_fuse_signal_scores({
        "folded_row": {"B": 1.0, "A": 1.0},
    })
    assert tie_ranks["A"]["folded_row"] == 1
    assert tie_ranks["B"]["folded_row"] == 2
    assert sorted(tie_totals, key=lambda table_id: (-tie_totals[table_id], table_id)) == ["A", "B"]

    signals = {
        "folded_row": {"B": 2.0, "A": 1.0},
        "folded_context": {"B": 2.0, "A": 1.0},
        "title": {"A": 3.0, "B": 1.0},
        "header": {"A": 3.0, "B": 1.0},
        "unit": {"A": 3.0, "B": 1.0},
    }
    totals, _, families = rank_fuse_signal_scores(signals)
    assert totals["B"] > totals["A"]
    assert set(families["A"]) == {"row", "context", "metadata"}


def test_rank_fusion_retrieval_exposes_rank_trace(tmp_path):
    path = tmp_path / "report.txt"
    path.write_text(
        "Bảng một\n"
        "<table><tr><td>Chỉ tiêu</td><td>2024</td></tr><tr><td>Lợi nhuận</td><td>42</td></tr></table>\n"
        "Bảng hai\n"
        "<table><tr><td>Chỉ tiêu</td><td>2024</td></tr><tr><td>Lơi nhuận</td><td>1</td></tr></table>",
        encoding="utf-8",
    )
    identity = ReportIdentity("R", "ABC", 2024, "consolidated", path.name)
    result = retrieve_rows(
        "Lợi nhuận năm 2024",
        {"years": [2024], "tickers": ["ABC"]},
        [Report(identity, path)],
        top_k=2,
        mode="rank-fusion",
    )
    assert result["mode"] == "rank-fusion"
    assert [table["table_id"] for table in result["tables"]] == ["R|2", "R|4"]
    trace = result["tables"][0]["field_scores"]
    assert set(trace["signal_ranks"]) >= {
        "folded_row", "unicode_row", "folded_context", "unicode_context",
    }
    assert set(trace["family_contributions"]) == {"row", "context", "metadata"}


def test_field_aware_mode_ranks_and_exposes_field_scores(tmp_path):
    path = tmp_path / "report.txt"
    path.write_text(
        "Bảng doanh thu Đơn vị: triệu đồng\n"
        "<table><tr><td>Chỉ tiêu</td><td>2024</td></tr><tr><td>Doanh thu thuần</td><td>42</td></tr></table>\n"
        "Bảng khác\n"
        "<table><tr><td>Chỉ tiêu</td><td>2024</td></tr><tr><td>Chi phí</td><td>1</td></tr></table>",
        encoding="utf-8",
    )
    identity = ReportIdentity("R", "ABC", 2024, "consolidated", path.name)
    result = retrieve_rows(
        "Doanh thu thuần năm 2024",
        {"years": [2024], "tickers": ["ABC"]},
        [Report(identity, path)],
        top_k=5,
        mode="field-aware",
    )
    assert result["mode"] == "field-aware"
    assert result["tables"]
    assert result["tables"][0]["table_id"] == "R|2"
    assert set(result["tables"][0]["field_scores"]) >= {"row", "title", "header", "unit", "phrase", "rrf"}
    assert result["tables"][0]["score"] >= result["tables"][-1]["score"]
    coverage_result = retrieve_rows(
        "Doanh thu thuần năm 2024",
        {"years": [2024], "tickers": ["ABC"]},
        [Report(identity, path)],
        top_k=5,
        mode="field-coverage",
    )
    assert coverage_result["tables"][0]["table_id"] == "R|2"


def test_experimental_empty_result_falls_back_to_baseline(tmp_path, monkeypatch):
    path = tmp_path / "report.txt"
    path.write_text(
        "Bảng doanh thu\n<table><tr><td>Chỉ tiêu</td><td>2024</td></tr><tr><td>Doanh thu</td><td>42</td></tr></table>",
        encoding="utf-8",
    )
    identity = ReportIdentity("R", "ABC", 2024, "consolidated", path.name)
    baseline = retrieve_rows("Doanh thu năm 2024", {"years": [2024]}, [Report(identity, path)])
    monkeypatch.setattr(retrieval_module, "field_aware_table_scores", lambda *args, **kwargs: {})
    result = retrieve_rows("Doanh thu năm 2024", {"years": [2024]}, [Report(identity, path)], mode="field-aware")
    assert result["experimental_fallback"] is True
    assert result["tables"] == baseline["tables"]


def test_experimental_error_falls_back_to_baseline(tmp_path, monkeypatch):
    path = tmp_path / "report.txt"
    path.write_text("Bảng\n<table><tr><td>Doanh thu</td><td>42</td></tr></table>", encoding="utf-8")
    identity = ReportIdentity("R", "ABC", 2024, "consolidated", path.name)
    baseline = retrieve_rows("Doanh thu năm 2024", {"years": [2024]}, [Report(identity, path)])
    def fail(*args, **kwargs):
        raise RuntimeError("research mode failure")
    monkeypatch.setattr(retrieval_module, "field_aware_table_scores", fail)
    result = retrieve_rows("Doanh thu năm 2024", {"years": [2024]}, [Report(identity, path)], mode="field-aware")
    assert result["experimental_fallback"] is True
    assert result["tables"] == baseline["tables"]


def test_reranker_representation_deduplicates_repeated_context(tmp_path):
    path = tmp_path / "report.txt"
    path.write_text("Bảng doanh thu\n<table><tr><td>2024</td></tr><tr><td>42</td></tr></table>", encoding="utf-8")
    table = report_tables(str(path), ReportIdentity("R", "ABC", 2024, "consolidated", path.name))[0]
    text = table_representation(table, 1)
    assert text.count("2024") == 1


def test_evidence_slots_reserve_distinct_requested_years(tmp_path):
    path_2024 = tmp_path / "r24.txt"
    path_2023 = tmp_path / "r23.txt"
    table = "<table><tr><td>Revenue</td><td>42</td></tr></table>"
    path_2024.write_text("Bảng\n" + table, encoding="utf-8")
    path_2023.write_text("Bảng\n" + table, encoding="utf-8")
    reports = [
        Report(ReportIdentity("R24", "ABC", 2024, "consolidated", path_2024.name), path_2024),
        Report(ReportIdentity("R23", "ABC", 2023, "consolidated", path_2023.name), path_2023),
    ]
    result = retrieve_rows("Revenue", {"years": [2024, 2023]}, reports, mode="evidence-slots", top_k=2)
    assert {table["report_id"] for table in result["tables"]} == {"R24", "R23"}


def test_evidence_slots_use_derived_comparison_years(tmp_path):
    path_2024 = tmp_path / "r24.txt"
    path_2023 = tmp_path / "r23.txt"
    table = "<table><tr><td>Revenue</td><td>42</td></tr></table>"
    path_2024.write_text("Bảng\n" + table, encoding="utf-8")
    path_2023.write_text("Bảng\n" + table, encoding="utf-8")
    reports = [
        Report(ReportIdentity("R24", "ABC", 2024, "consolidated", path_2024.name), path_2024),
        Report(ReportIdentity("R23", "ABC", 2023, "consolidated", path_2023.name), path_2023),
    ]
    result = retrieve_rows(
        "Revenue growth", {"years": [2024], "slot_years": [2024, 2023]}, reports,
        mode="evidence-slots", top_k=2,
    )
    assert {table["report_id"] for table in result["tables"]} == {"R24", "R23"}


def test_report_coverage_reserves_one_table_per_gated_report(tmp_path):
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("Bảng\n<table><tr><td>Revenue</td><td>42</td></tr></table>", encoding="utf-8")
    second.write_text("Bảng\n<table><tr><td>Revenue</td><td>41</td></tr></table>", encoding="utf-8")
    reports = [
        Report(ReportIdentity("R1", "AAA", 2024, "consolidated", first.name), first),
        Report(ReportIdentity("R2", "BBB", 2024, "consolidated", second.name), second),
    ]
    result = retrieve_rows("Revenue", {"years": [2024]}, reports, mode="report-coverage", top_k=2)
    assert {table["report_id"] for table in result["tables"]} == {"R1", "R2"}
    metric_result = retrieve_rows("Tính tỷ lệ Revenue", {"years": [2024]}, reports, mode="metric-coverage", top_k=2)
    assert {table["report_id"] for table in metric_result["tables"]} == {"R1", "R2"}


def test_table_budget_scales_with_gated_reports_and_honors_explicit_values():
    assert table_budget(1) == 3
    assert table_budget(4) == 12
    assert table_budget(0) == 1
    assert table_budget(40) == 30
    assert table_budget(4, 5) == 5
    assert table_budget(4, "5") == 5


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


def test_v2_template_normalization_removes_entity_year_and_numbers():
    first = "Doanh thu (ABC) năm 2024 là 42,5%?"
    second = "Doanh thu (XYZ) năm 2019 là 7%?"
    assert normalize_template(first) == normalize_template(second)


def test_v2_sampler_is_deterministic_and_prevents_sealed_leakage():
    records = []
    item_id = 1
    for ticker in ("AAA", "BBB", "CCC", "DDD"):
        for year in range(2016, 2024):
            records.append({
                "id": item_id,
                "question": f"Chi tieu metric{chr(97 + item_id % 20)} cua cong ty ({ticker}) nam {year} la bao nhieu?",
            })
            item_id += 1
    frame = build_frame(records)
    first = sample_splits(frame, {1, 2}, development_count=4, core_count=8, ood_count=5, seed=9)
    second = sample_splits(frame, {1, 2}, development_count=4, core_count=8, ood_count=5, seed=9)
    assert manifest(first, seed=9) == manifest(second, seed=9)
    ids = {name: {item.id for item in values} for name, values in first.items()}
    assert not ids["development"] & ids["sealed_core"]
    assert not ids["development"] & ids["sealed_ood"]
    sealed = first["sealed_core"] + first["sealed_ood"]
    assert not {item.template_hash for item in first["development"]} & {item.template_hash for item in sealed}
    assert not {item.report_key for item in first["development"]} & {item.report_key for item in sealed}


def test_v2_sampler_rejects_non_question_inputs():
    try:
        build_frame([{"id": 1, "question": "Q", "prediction": ["table"]}])
    except ValueError as error:
        assert "question id and text" in str(error)
    else:
        raise AssertionError("sampler accepted a prediction-bearing record")


def _v2_annotation():
    return {
        "schema_version": "2.0", "id": 7, "question_hash": "q", "complete": True,
        "operation": "lookup", "confidence": 0.9,
        "reviewer_protocol_hash": "reviewer-a", "slots": [{
            "slot_id": "entity_year_metric_value", "entity": "ABC", "report_year": 2024,
            "scope": "unknown", "metric": "revenue", "operand_role": "value",
            "alternatives": [
                {"reports": ["R"], "tables": ["A", "B"], "cells": []},
                {"reports": ["R"], "tables": ["C", "D"], "cells": []},
            ],
        }],
    }


def test_v2_slot_metrics_choose_complete_alternative_not_flat_union():
    annotation = _v2_annotation()
    assert validate_v2_annotation(annotation) == []
    mixed = score_slots(annotation, ["A", "C"], k=2)
    assert mixed["slot_recall"] == 0.5
    assert mixed["question_complete"] is False
    accepted = score_slots(annotation, ["C", "D"], k=2)
    assert accepted["question_complete"] is True
    partial = score_slots(annotation, ["A", "X"], k=2)
    assert partial["slot_recall"] == 0.5
    assert partial["question_complete"] is False
    assert partial["mrr"] == 1.0


def test_v2_slot_metrics_mark_over_budget_evidence_infeasible():
    annotation = _v2_annotation()
    annotation["slots"][0]["alternatives"] = [{
        "reports": ["R"], "tables": ["A", "B", "C", "D", "E", "F"], "cells": [],
    }]
    assert score_slots(annotation, ["A", "B", "C", "D", "E"], k=5)["feasible_coverage"] is False


def test_v2_slot_metrics_consider_total_question_budget():
    annotation = _v2_annotation()
    annotation["slots"].append({
        "slot_id": "second", "entity": "ABC", "report_year": 2024, "scope": "unknown",
        "metric": "cost", "operand_role": "value",
        "alternatives": [{"reports": ["R"], "tables": ["E", "F", "G"], "cells": []}],
    })
    annotation["slots"][0]["alternatives"] = [{
        "reports": ["R"], "tables": ["A", "B", "C"], "cells": [],
    }]
    assert score_slots(annotation, ["A", "B", "C", "E", "F"], k=5)["feasible_coverage"] is False


def test_v2_adjudication_requires_two_independent_reviews_and_resolution():
    adjudicated = _v2_annotation()
    adjudicated["adjudication"] = {"resolved": True}
    reviewer_b = _v2_annotation()
    reviewer_b["reviewer_protocol_hash"] = "reviewer-b"
    assert validate_adjudication(adjudicated, _v2_annotation(), reviewer_b) == []
    assert "both independent reviewer files are required" in validate_adjudication(adjudicated, None, reviewer_b)


def test_v2_batch_validation_checks_raw_source_coordinates(tmp_path):
    report_id = "ABC_financial_statements_2024_consolidated"
    source = tmp_path / "financial_statements" / "ABC" / "2024" / report_id / f"{report_id}_extracted.txt"
    source.parent.mkdir(parents=True)
    source.write_text("<table><tr><td>Metric</td><td>42</td></tr></table>", encoding="utf-8")
    adjudicated = _v2_annotation()
    adjudicated["adjudication"] = {"resolved": True}
    adjudicated["slots"][0]["alternatives"] = [{
        "reports": [report_id], "tables": [f"{report_id}|1"],
        "cells": [{"table": f"{report_id}|1", "row": 0, "column": 1, "raw": "42", "period": "unknown", "unit": "unknown"}],
    }]
    reviewer_a = json.loads(json.dumps(adjudicated))
    reviewer_b = json.loads(json.dumps(adjudicated))
    reviewer_b["reviewer_protocol_hash"] = "reviewer-b"
    assert validate_annotation_batch([adjudicated], [reviewer_a], [reviewer_b], tmp_path) == []
    adjudicated["slots"][0]["alternatives"][0]["cells"][0]["raw"] = "wrong"
    assert "raw cell mismatch" in validate_annotation_batch([adjudicated], [reviewer_a], [reviewer_b], tmp_path)[0]


def test_v2_weighted_evaluation_bootstrap_and_immutable_artifacts(tmp_path):
    annotation = _v2_annotation()
    manifest_data = {"splits": {"sealed_core": [{
        "id": 7, "report_key": ["ABC", 2024, "unknown"], "inclusion_probability": 0.5,
    }]}}
    baseline = evaluate_v2_predictions([annotation], [{"id": 7, "ranked_tables": ["A"]}], manifest_data, split="sealed_core")
    candidate = evaluate_v2_predictions([annotation], [{"id": 7, "ranked_tables": ["C", "D"], "latency_ms": 2.0}], manifest_data, split="sealed_core")
    assert summarize_v2_traces(candidate, weighted=True)["question_complete"] == 1.0
    interval = paired_cluster_bootstrap(baseline, candidate, iterations=20, seed=1)
    assert interval["question_complete"]["delta"] == 1.0
    input_file = tmp_path / "input.json"
    input_file.write_text("{}", encoding="utf-8")
    output = tmp_path / "run"
    atomic_write_run(output, {"summary.json": {"ok": True}}, {"input": input_file})
    assert (output / "manifest.json").is_file()
    try:
        atomic_write_run(output, {"summary.json": {"ok": True}}, {"input": input_file})
    except FileExistsError:
        pass
    else:
        raise AssertionError("immutable run overwrote existing output")


def test_v2_predictions_reject_duplicate_table_ids():
    annotation = _v2_annotation()
    manifest_data = {"splits": {"sealed_core": [{"id": 7, "report_key": ["ABC", 2024, "unknown"], "inclusion_probability": 0.5}]}}
    try:
        evaluate_v2_predictions([annotation], [{"id": 7, "ranked_tables": ["A", "A"]}], manifest_data, split="sealed_core")
    except ValueError as error:
        assert "duplicate table IDs" in str(error)
    else:
        raise AssertionError("duplicate ranked table IDs accepted")


def test_v2_promotion_gate_accepts_only_all_boundary_conditions():
    summary = {"records": 1, "latency_ms": {"p95": 10.0}}
    bootstrap = {metric: {"delta": 0.0, "ci95_low": 0.0} for metric in ("slot_recall", "mrr", "ndcg")}
    bootstrap["slot_recall"] = {"delta": 0.1, "ci95_low": 0.001}
    trace = [{"fallback": False, "source_binding_valid": True, "top_five_valid": True}]
    assert promotion_gate(summary, summary, bootstrap, trace)["passed"] is True
    bootstrap["slot_recall"]["ci95_low"] = 0.0
    assert "slot_recall_ci95_low_not_positive" in promotion_gate(summary, summary, bootstrap, trace)["failures"]
