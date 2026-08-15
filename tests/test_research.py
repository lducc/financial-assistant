"""Focused tests for OCR parsing, retrieval, and pilot source bindings."""

import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifinqa.retrieval import Report, metric_query_tokens, rank_fuse_signal_scores, report_tables, retrieve_rows, table_budget
from vifinqa.answers import EvidenceValue, answer_plan, first_numeric_cell, fold, operator_text, parse_ocr_number, says
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


def load_script(name: str):
    """Import a scripts/ entry point that is not part of an installed package."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def load_kaggle(name: str, **environment):
    """Import a kaggle/ notebook script, whose settings are read at import time."""
    import os

    previous = {key: os.environ.get(key) for key in environment}
    os.environ.update(environment)
    try:
        spec = importlib.util.spec_from_file_location(name, ROOT / "kaggle" / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        assert spec.loader
        spec.loader.exec_module(module)
        return module
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


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
    # Percentages come back as percentage points, rounded to two decimals.
    answer, _ = answer_plan("Tỷ lệ phần trăm là bao nhiêu?", values) or (None, "")
    assert answer == 33.33


def test_answer_plan_converts_to_the_unit_the_question_asks_for():
    billions = [EvidenceValue("df0", 1, 3, 5_120_000_000.0, "R1")]
    answer, expression = answer_plan("Tiền và tương đương tiền là bao nhiêu tỷ đồng?", billions) or (None, "")
    assert answer == 5.12 and "/ 1000000000.0" in expression
    answer, _ = answer_plan("... là bao nhiêu triệu đồng?", billions) or (None, "")
    assert answer == 5120.0


def test_emitted_query_reproduces_the_answer_through_pandas(tmp_path):
    """Execute the emitted Pandas the way the evaluator does: read_csv, then iloc."""
    import csv as csv_module

    import pandas as pd

    grid = [
        ["", "2018VND", "2017VND"],
        ["Lãi tiền gửi", "208.253.201.298", "69.917.578.051"],
        ["Lãi khác", "85.422.296.361", "43.977.690.600"],
    ]
    path = tmp_path / "table.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv_module.writer(handle).writerows(grid)

    # Row 1 of the parsed grid is the figure we want; column 1 is the 2018 column.
    value = EvidenceValue("df0", 1, 1, 208_253_201_298.0, "R")
    answer, expression = answer_plan("Lãi tiền gửi năm 2018 là bao nhiêu triệu đồng?", [value])
    assert answer == 208253.2

    namespace = {"df0": pd.read_csv(path)}
    exec(f"result = {expression}", namespace)  # noqa: S102 - mirrors the evaluator
    # Compared raw. An earlier version of this test rounded the result before
    # comparing, which is exactly what the query failed to do, so it passed while
    # 231 of 1,012 submitted queries returned a figure 0.2% from their own answer
    # against a 0.02% tolerance.
    assert namespace["result"] == answer


def test_emitted_query_rounds_the_way_the_answer_does(tmp_path):
    """A figure that does not round cleanly is where answer and query diverge."""
    import csv as csv_module

    import pandas as pd

    grid = [["", "2018VND"], ["Doanh thu", "1.422.900.000"]]
    path = tmp_path / "table.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv_module.writer(handle).writerows(grid)

    value = EvidenceValue("df0", 1, 1, 1_422_900_000.0, "R")
    answer, expression = answer_plan("Doanh thu năm 2018 là bao nhiêu tỷ đồng?", [value])
    assert answer == 1.42

    namespace = {"df0": pd.read_csv(path)}
    exec(f"result = {expression}", namespace)  # noqa: S102 - mirrors the evaluator
    # Unrounded this returns 1.4229, which is 0.20% out — ten times the tolerance.
    assert namespace["result"] == answer


def test_contents_page_is_not_a_candidate():
    """The index page is the best lexical match a question will ever find.

    Its row labels are the names of the statements, so a question about cash flow
    ranks it first, and the answer path then reads a page number as the figure.
    It cost 57 submitted tables across 32 questions, none of which could be right.
    """
    from vifinqa.retrieval import Table, is_contents_page

    def table(rows):
        return Table(
            "R|1", "R", 1, 1, tuple(tuple(row) for row in rows), "t", (), (), (), "",
        )

    assert is_contents_page(table([
        ["Bảng cân đối kế toán hợp nhất", "6 - 8"],
        ["Báo cáo lưu chuyển tiền tệ hợp nhất", "10 - 11"],
        ["Thuyết minh báo cáo tài chính", "12 - 57"],
    ]))
    # The header form, where a column is literally named "Trang".
    assert is_contents_page(table([["Chỉ tiêu", "Trang"], ["Bảng cân đối kế toán", "9"]]))
    # A statement is not rejected, and neither is a table of small ratios, which
    # would be the obvious false positive for a rule keyed on figure size.
    assert not is_contents_page(table([
        ["Doanh thu", "208.253.201.298", "69.917.578.051"],
        ["Giá vốn", "1.422.900.000", "1.000.000"],
    ]))
    assert not is_contents_page(table([["ROE", "15,2", "14,8"], ["ROA", "5,1", "4,9"]]))


def test_per_item_only_changes_questions_that_name_several_items():
    """The control and the treatment must differ on exactly the multi-item half.

    A question naming zero or one line item has to produce a byte-identical
    prompt under both settings, or the full-corpus comparison against
    scores_v4.jsonl is confounded by questions that were never meant to move.
    """
    off = load_kaggle("rerank_qwen_8b", PER_ITEM="0")
    on = load_kaggle("rerank_qwen_8b", PER_ITEM="1")
    assert not off.PER_ITEM and on.PER_ITEM

    bare = {"question": "Doanh thu 2024?", "line_items": []}
    single = {"question": "Doanh thu 2024?", "line_items": ["doanh thu"]}
    double = {"question": "Doanh thu và giá vốn 2024?", "line_items": ["doanh thu", "giá vốn"]}

    assert on.build_queries(bare) == off.build_queries(bare)
    assert on.build_queries(single) == off.build_queries(single)
    assert len(off.build_queries(double)) == 1
    assert len(on.build_queries(double)) == 2
    # The joined query names both items, so it is a different string entirely.
    assert on.build_queries(double) != off.build_queries(double)


def test_per_item_writes_a_matrix_whose_max_is_the_shipped_score():
    """The matrix is additive: `scores` must stay exactly what it was."""
    module = load_kaggle("rerank_qwen_8b", PER_ITEM="1")
    record = {
        "id": 7,
        "question": "Doanh thu và giá vốn 2024?",
        "line_items": ["doanh thu", "giá vốn"],
        "candidates": [
            {"table_id": "R|1", "sparse_rank": 1, "text": "a"},
            {"table_id": "R|2", "sparse_rank": 2, "text": "b"},
            # Past RERANK_DEPTH, so it is left to the sparse order entirely.
            {"table_id": "R|3", "sparse_rank": module.RERANK_DEPTH + 1, "text": "c"},
        ],
    }
    matrix = {0: [0.9, 0.1], 1: [0.2, 0.8]}
    values = [0.9, 0.8, 0.0]
    judged = module.to_judge(record)
    assert judged == [0, 1], "the third candidate is past RERANK_DEPTH"

    payload = module.score_payload(record, values, matrix, judged, per_item=True)
    assert payload["scores"] == {"R|1": 0.9, "R|2": 0.8}
    assert payload["line_items"] == ["doanh thu", "giá vốn"]
    assert payload["per_item"] == {"R|1": [0.9, 0.1], "R|2": [0.2, 0.8]}
    for table_id, row in payload["per_item"].items():
        assert max(row) == payload["scores"][table_id]

    # With the flag off the line is what every committed score file already is.
    assert module.score_payload(record, values, matrix, judged, per_item=False) == {
        "id": 7, "scores": {"R|1": 0.9, "R|2": 0.8},
    }
    # A question naming no item has nothing to decompose, so no matrix is written.
    bare = {**record, "line_items": []}
    assert "per_item" not in module.score_payload(
        bare, values, {0: [0.9], 1: [0.8]}, judged, per_item=True
    )


def test_operator_routing_ignores_line_item_words():
    values = [EvidenceValue("df0", 1, 2, 10.0, "R1"), EvidenceValue("df1", 2, 3, 30.0, "R2")]
    # "Tổng phải thu ngắn hạn khác" is a line item, not an instruction to add.
    text = operator_text("Tổng phải thu ngắn hạn khác của OGC đến ngày 31/12/2019?")
    assert "phai thu ngan han khac" not in text
    # And "bao nhiêu" must never read as "hiệu" once diacritics are folded.
    assert not says(fold("Doanh thu là bao nhiêu?"), "hieu")
    assert says(fold("Chênh lệch giữa hai năm?"), "chenh lech")


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
    # Four reciprocal-rank terms contribute here — row, context, metric, and
    # supporting rows — so the winner scores 4/(60+1) and the runner-up 1/(60+2).
    assert [
        (table["score"], table["row_index"], table["row_cells"])
        for table in result["tables"]
    ] == [
        (0.065574, 1, ["Doanh thu", "42"]),
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


def test_question_tiers_follow_the_published_difficulty_definitions():
    classify = load_script("classify_questions").classify
    assert classify("Tiền và tương đương tiền của VJC cuối năm 2018 là bao nhiêu?", ["VJC"], [2018]) == "easy"
    assert classify(
        "Tăng trưởng doanh thu thuần năm 2024 so với 2023 của VJC là bao nhiêu?", ["VJC"], [2023, 2024],
    ) == "medium"
    assert classify(
        "Doanh thu thuần trung bình của VJC giai đoạn 2020-2024 là bao nhiêu?", ["VJC"], [2020, 2021, 2022, 2023, 2024],
    ) == "intermediate"
    assert classify(
        "Trong nhóm HPG, HSG và NKG, doanh nghiệp có doanh thu cao nhất năm 2024 ghi nhận hàng tồn kho bao nhiêu?",
        ["HPG", "HSG", "NKG"], [2024],
    ) == "hard"


def test_unused_evidence_variables_match_whole_names_not_prefixes():
    import re

    expression = "float(df10.iloc[0, 1]) + float(df2.iloc[0, 1])"
    # "df1" is a prefix of "df10"; a containment test would call it referenced and
    # the strict submission validator would then reject the package.
    unused = [f"df{rank}" for rank in range(12) if not re.search(rf"\bdf{rank}\b", expression)]
    assert "df1" in unused
    assert "df10" not in unused and "df2" not in unused


def test_diagnostics_attribute_each_miss_to_the_stage_that_lost_it():
    classify = load_script("diagnose_retrieval").classify
    trace = {
        "gold_tables": ["R1|10", "R1|20", "R2|30", "R3|40"],
        "ranked_tables": ["R1|10", "R9|99", "R1|20"],
        "selected_docs": ["R1", "R2"],
    }
    # Budget of one: the first gold table ships, the second is retrieved but below
    # budget, the third is gated but never retrieved, the fourth was never gated.
    states = classify(trace, budget=1)
    assert states["hit"] == ["R1|10"]
    assert states["rank_miss"] == ["R1|20"]
    assert states["candidate_miss"] == ["R2|30"]
    assert states["gate_miss"] == ["R3|40"]


def test_benchmark_records_carry_verifiable_bindings():
    benchmark = ROOT / "annotations" / "benchmark.jsonl"
    records = [json.loads(line) for line in benchmark.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(records) == len({record["id"] for record in records})
    for record in records:
        annotation = record["annotation"]
        assert annotation["gold_tables"], record["id"]
        assert record["source"] in {"gold_150", "v3"}
        assert record["taxonomy"]["table_count"] == len(annotation["gold_tables"])
        assert sorted({table.partition("|")[0] for table in annotation["gold_tables"]}) == annotation["gold_reports"]
        # Bindings prove the label; a record without them cannot be rechecked.
        assert annotation["row_column_bindings"], record["id"]


def test_metric_phrase_strips_question_boilerplate_but_keeps_the_line_item():
    phrase = load_script("propose_evidence").metric_phrase(
        "Số dư phải thu phí quản lý tập trung của công ty mẹ Tập đoàn Công nghiệp Cao su Việt Nam - CTCP "
        "cuối năm 2020 là bao nhiêu triệu đồng?",
        ["GVR"],
    )
    assert phrase == "so du phai thu phi quan ly tap trung"


def test_cross_validation_folds_never_split_a_report_cluster():
    module = load_script("cross_validate_retrieval")
    labels = [
        {"id": 1, "annotation": {"gold_reports": ["A", "B"]}},
        {"id": 2, "annotation": {"gold_reports": ["B"]}},
        {"id": 3, "annotation": {"gold_reports": ["C"]}},
    ]
    cluster_by_report = {
        report: group[0] for group in module.connected_report_groups(labels) for report in group
    }
    # Questions 1 and 2 share report B, so they must land in the same fold.
    clusters = {
        record["id"]: cluster_by_report[record["annotation"]["gold_reports"][0]] for record in labels
    }
    assert clusters[1] == clusters[2] != clusters[3]
    assert module.fold_of(clusters[1], 5) == module.fold_of(clusters[2], 5)


def test_table_budget_scales_with_gated_reports_and_honors_explicit_values():
    assert table_budget(1) == 2
    assert table_budget(4) == 8
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
    module = load_script("evaluate_table_retrieval")
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


def test_binding_gold_narrows_to_bound_tables_and_is_the_default():
    """The gold set that reproduces live is the one named by row/column bindings.

    Restatement completion widened gold from 2.57 to 4.50 tables per question,
    but submitting k tables against G gold gives precision/recall = G/k and live
    reads 0.563, which the bound tables match and the widened set does not. The
    narrow definition therefore has to be what scoring uses unless asked
    otherwise.
    """
    evaluate = load_script("evaluate_table_retrieval")
    annotation = {
        "gold_tables": ["r|1", "r|2", "r|3"],
        "gold_reports": ["r"],
        "row_column_bindings": [
            {"table": "r|1", "row": 4, "column": 2},
            {"table": "r|1", "row": 5, "column": 2},
            {"table": "r|3", "row": 9, "column": 2},
        ],
    }
    assert evaluate.gold_tables_for(annotation, "full") == ["r|1", "r|2", "r|3"]
    # Deduplicated, and "r|2" carries no binding so it is a restatement.
    assert evaluate.gold_tables_for(annotation, "binding") == ["r|1", "r|3"]
    assert evaluate.gold_tables_for(annotation) == ["r|1", "r|3"]
    record = {"id": 1, "question": "q", "annotation": annotation}
    assert evaluate.score_record(record, ["r|1"], ["r"])["gold_tables"] == ["r|1", "r|3"]


def test_binding_gold_falls_back_rather_than_scoring_against_nothing():
    """A record with no bindings keeps its wide gold set.

    Narrowing it to the empty set would score the question as a total miss no
    matter what was retrieved, silently deflating every aggregate it enters.
    """
    evaluate = load_script("evaluate_table_retrieval")
    annotation = {"gold_tables": ["r|1"], "gold_reports": ["r"], "row_column_bindings": []}
    assert evaluate.gold_tables_for(annotation, "binding") == ["r|1"]
    assert evaluate.score_record(
        {"id": 1, "question": "q", "annotation": annotation}, ["r|1"], ["r"],
    )["prefix"]["1"]["recall"] == 1.0


def test_cached_traces_predating_binding_gold_still_score():
    """Traces cached before the narrow definition existed carry only the wide set."""
    cv = load_script("cross_validate_retrieval")
    fresh = {"gold_tables": ["r|1", "r|2"], "gold_tables_binding": ["r|1"], "ranked_tables": ["r|1"]}
    stale = {"gold_tables": ["r|1", "r|2"], "ranked_tables": ["r|1"]}
    assert cv.gold_of(fresh, "binding") == ["r|1"]
    assert cv.gold_of(fresh, "full") == ["r|1", "r|2"]
    assert cv.gold_of(stale, "binding") == ["r|1", "r|2"]
    assert cv.score(fresh, 1, "binding")["recall"] == 1.0
    assert cv.score(fresh, 1, "full")["recall"] == 0.5


def test_submission_queries_need_not_read_every_submitted_table(tmp_path):
    """A query reads the cells it needs; padding to touch all of them is noise.

    The organizers reject a constant result and read the queries by hand in the
    private phase, so a query must compute from a submitted table — but a
    question needing one figure out of six retrieved tables legitimately touches
    one. Requiring all of them appended "+ 0 * dfN.shape[0]" to 97% of rows,
    terms that compute nothing on the very queries a reviewer inspects.
    """
    validate = load_script("validate_submission")

    def package(query: str) -> Path:
        root = tmp_path / query[:12].replace(" ", "_").replace("=", "")
        (root / "data" / "tables").mkdir(parents=True, exist_ok=True)
        for name in ("t0", "t1"):
            (root / "data" / "tables" / f"{name}.csv").write_text("a,b\n1,2\n", encoding="utf-8")
        (root / "submission.json").write_text(json.dumps([{
            "id": 1, "question": "q", "answer": 1.0,
            "relevant_docs": ["r"], "relevant_tables": ["r|1", "r|2"],
            "evidence": [{"variable": "df0", "csv_path": "data/tables/t0.csv"},
                         {"variable": "df1", "csv_path": "data/tables/t1.csv"}],
            "pandas_query": query,
        }]), encoding="utf-8")
        return root

    reads_one = validate.validate(package("result = float(df0.iloc[3, 1])"))
    assert not [error for error in reads_one if "evidence variable" in error], reads_one
    reads_none = validate.validate(package("result = 1234.0"))
    assert reads_none, "a query reading no submitted table must still be rejected"


def test_listwise_parsing_never_loses_a_candidate():
    """A generated permutation may be malformed; it may not shorten the ranking.

    A candidate dropped while parsing disappears from the submission and takes
    its retrieval score with it, so every failure mode has to degrade to the
    incoming order rather than to a shorter one.
    """
    from vifinqa.listwise import parse_permutation, reorder
    candidates = ["a", "b", "c", "d"]
    assert reorder(candidates, "[3] > [1] > [4] > [2]") == ["c", "a", "d", "b"]
    # Repeats keep the first mention; a second one is not a reordering signal.
    assert reorder(candidates, "[2] > [2] > [1]") == ["b", "a", "c", "d"]
    # Out of range labels have no candidate to attach to.
    assert reorder(candidates, "[9] > [3]") == ["c", "a", "b", "d"]
    # Truncated, empty and junk output all fall back to the incoming order.
    assert reorder(candidates, "[2] >") == ["b", "a", "c", "d"]
    assert reorder(candidates, "") == candidates
    assert reorder(candidates, "I cannot rank these") == candidates
    for text in ("[3] > [1]", "", "[9]", "[1] [1] [1]", "garbage"):
        assert sorted(reorder(candidates, text)) == sorted(candidates)
    assert parse_permutation("[1] [5]", 4) == [0]


def test_listwise_splice_keeps_the_tail_and_admits_nothing_new():
    """Only retrieved tables may be submitted, whatever the model emitted."""
    from vifinqa.listwise import splice
    ranking = ["a", "b", "c", "d", "e"]
    assert splice(ranking, ["c", "a"]) == ["c", "a", "b", "d", "e"]
    # A hallucinated table id is ignored rather than inserted.
    assert splice(ranking, ["c", "zzz", "a"]) == ["c", "a", "b", "d", "e"]
    assert splice(ranking, []) == ranking
    assert sorted(splice(ranking, ["e", "d", "c", "b", "a"])) == sorted(ranking)


def test_borda_needs_agreement_across_passes():
    """A candidate must place well under every presentation, not just one.

    The point of ranking the same window twice is that an LLM is biased toward
    the order it was shown; aggregating has to reward consistency, or the second
    pass adds nothing.
    """
    from vifinqa.listwise import borda
    # Unanimous orders survive unchanged.
    assert borda([["a", "b", "c"], ["a", "b", "c"]]) == ["a", "b", "c"]
    # Two exactly reversed passes tie every candidate by construction, so there
    # is no information to act on and the incoming order stands. That is the
    # behaviour we want when the model contradicts itself completely.
    assert borda([["a", "b", "c"], ["c", "b", "a"]]) == ["a", "b", "c"]
    # Partial disagreement does move things: "b" is second and first, "a" is
    # first and last, so consistency wins.
    assert borda([["a", "b", "c"], ["b", "c", "a"]])[0] == "b"
    # One order in, one order out.
    assert borda([["b", "a"]]) == ["b", "a"]
    assert borda([]) == []
    assert borda([[], []]) == []
    # An omission counts as last, not as absent: "a" leads one pass and is
    # missing from the other, so it keeps a middle place rather than winning.
    assert borda([["a", "b", "c"], ["b", "c"]]) == ["b", "a", "c"]
    # Each pass carries equal weight however long it is, so a short second pass
    # cannot be outvoted merely by being short.
    assert borda([["a", "b", "c", "d"], ["d", "a"]])[0] == "a"
    # Nothing is created or lost.
    merged = borda([["a", "b", "c"], ["c", "a", "b"]])
    assert sorted(merged) == ["a", "b", "c"]


def test_agreement_separates_a_judging_model_from_a_drifting_one():
    """The diagnostic that decides what a flat result means."""
    from vifinqa.listwise import agreement
    assert agreement(["a", "b", "c", "d"], ["a", "b", "c", "d"]) == 1.0
    assert agreement(["a", "b", "c", "d"], ["d", "c", "b", "a"]) == -1.0
    assert abs(agreement(["a", "b", "c", "d"], ["a", "c", "b", "d"])) < 1.0
    # Degenerate inputs must not raise.
    assert agreement(["a"], ["a"]) == 1.0
    assert agreement([], []) == 1.0


def test_scoring_a_ranking_reorders_candidates_without_admitting_any():
    """A reranker permutes what retrieval found; it cannot retrieve.

    If a ranking could introduce a table the trace never held, `candidate_miss`
    would silently turn into a ranking success and the attribution that decides
    where to work would be wrong.
    """
    diagnose = load_script("diagnose_retrieval")
    trace = {"id": 7, "ranked_tables": ["r|1", "r|2", "r|3"], "selected_docs": ["r"]}
    assert diagnose.reorder(trace, {"7": ["r|3", "r|1", "r|2"]})["ranked_tables"] == ["r|3", "r|1", "r|2"]
    # A table the ranking names but retrieval never surfaced stays out.
    assert diagnose.reorder(trace, {"7": ["r|9", "r|2"]})["ranked_tables"] == ["r|2", "r|1", "r|3"]
    # Candidates the ranking omits keep their retrieved order behind the named ones.
    assert diagnose.reorder(trace, {"7": ["r|3"]})["ranked_tables"] == ["r|3", "r|1", "r|2"]
    # A question the ranking does not cover is scored exactly as retrieved.
    assert diagnose.reorder(trace, {"8": ["r|3"]}) is trace
    assert diagnose.reorder(trace, {})["ranked_tables"] == trace["ranked_tables"]


def test_score_files_union_within_a_question_not_just_across_them(tmp_path):
    """A deeper export shares questions with the shallower one it extends.

    Sharded Kaggle runs never share a question, so replacing one file's dict with
    another's was correct. A depth-100 delta run writes only the candidates the
    depth-50 file lacks, and replacing there would throw away the first 50.
    """
    apply_scores = load_script("apply_rerank_scores")
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text(json.dumps({"id": 1, "candidates": [
        {"table_id": "r|1", "sparse_rank": 1},
        {"table_id": "r|2", "sparse_rank": 2},
        {"table_id": "r|3", "sparse_rank": 3},
    ]}) + "\n", encoding="utf-8")
    shallow = tmp_path / "shallow.jsonl"
    shallow.write_text(json.dumps({"id": 1, "scores": {"r|1": 0.1, "r|2": 0.2}}) + "\n", encoding="utf-8")
    deep = tmp_path / "deep.jsonl"
    deep.write_text(json.dumps({"id": 1, "scores": {"r|3": 0.9}}) + "\n", encoding="utf-8")

    output = tmp_path / "ranking.json"
    sys.argv = ["apply", "--pairs", str(pairs), "--scores", str(shallow), "--scores", str(deep),
                "--output", str(output), "--mode", "replace"]
    apply_scores.main()
    # All three are scored, so replace orders all three by score. Dropping the
    # shallow file would leave r|1 and r|2 unscored and trailing in sparse order.
    assert json.loads(output.read_text(encoding="utf-8"))["1"] == ["r|3", "r|2", "r|1"]


def test_a_deeper_run_skips_the_pairs_a_previous_one_already_judged(tmp_path):
    """The skipped pairs must be absent, not zero.

    A zero is a score. Writing one for a candidate this run did not judge would
    rank it below everything the model actively disliked, instead of leaving it
    to the earlier file that already holds its score.
    """
    module = load_kaggle("rerank_qwen_8b")
    record = {
        "id": 1,
        "question": "Doanh thu 2024?",
        "line_items": ["doanh thu"],
        "candidates": [
            {"table_id": "r|1", "sparse_rank": 1, "text": "a"},
            {"table_id": "r|2", "sparse_rank": 2, "text": "b"},
            {"table_id": "r|3", "sparse_rank": 3, "text": "c"},
        ],
    }
    assert module.to_judge(record) == [0, 1, 2]
    assert module.to_judge(record, {"r|1", "r|2"}) == [2]

    prior = tmp_path / "prior.jsonl"
    prior.write_text(json.dumps({"id": 1, "scores": {"r|1": 0.4, "r|2": 0.5}}) + "\n", encoding="utf-8")
    assert module.already_judged(str(prior)) == {1: {"r|1", "r|2"}}
    assert module.already_judged(None) == {}

    payload = module.score_payload(record, [0.0, 0.0, 0.7], {2: [0.7]}, [2], per_item=False)
    assert payload == {"id": 1, "scores": {"r|3": 0.7}}


def test_comparing_two_runs_reports_the_drift_beside_the_effect(tmp_path):
    """The A/A stratum is the floor the treated stratum has to clear."""
    compare = load_script("compare_rerank_runs")
    control = {1: {"r|1": 0.5, "r|2": 0.25}}
    treatment = {1: {"r|1": 0.5, "r|2": 0.75}}
    assert compare.agreement(control, treatment) == {
        "shared_pairs": 2, "identical": 1, "identical_share": 0.5,
        "mean_abs_delta": 0.25, "max_abs_delta": 0.5,
    }

    # Questions naming at most --aa-items line items have an identical prompt
    # under both settings, so they measure drift; the rest are the treated half.
    pairs = {1: {"line_items": []}, 2: {"line_items": ["a"]}, 3: {"line_items": ["a", "b"]}}
    assert compare.strata(pairs, [1, 2, 3], 1) == ([1, 2], [3])
    # A change that reaches every question makes the whole set an A/A, which is
    # how drift is measured on purpose rather than inferred.
    assert compare.strata(pairs, [1, 2, 3], 99) == ([1, 2, 3], [])


def test_fusion_mode_can_be_chosen_per_difficulty_tier(tmp_path):
    """Easy questions keep the sparse prior; hard ones take the model's order.

    BM25 matches an easy question's one table by label and its rank is real
    information; on a hard question spanning many gated reports it is close to
    arbitrary, and fusing it only dilutes the cross-encoder.
    """
    apply_scores = load_script("apply_rerank_scores")
    candidates = [
        {"table_id": "r|1", "sparse_rank": 1},
        {"table_id": "r|2", "sparse_rank": 2},
        {"table_id": "r|3", "sparse_rank": 3},
    ]
    scores = {"r|1": 0.1, "r|2": 0.2, "r|3": 0.9}
    # The model's order alone.
    assert apply_scores.fuse(candidates, scores, "replace") == ["r|3", "r|2", "r|1"]
    # Fused, the sparse rank keeps the model's third choice from jumping the queue.
    assert apply_scores.fuse(candidates, scores, "fuse")[0] in {"r|1", "r|3"}

    # The weight interpolates between the two orders rather than switching, so the
    # endpoints have to agree with the modes they generalize.
    assert apply_scores.fuse(candidates, scores, "fuse", 0.0) == ["r|3", "r|2", "r|1"]
    assert apply_scores.fuse(candidates, scores, "fuse", 1.0) == ["r|1", "r|2", "r|3"]

    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text("".join(
        json.dumps({"id": identifier, "candidates": candidates}, ensure_ascii=False) + "\n"
        for identifier in (1, 2)
    ), encoding="utf-8")
    score_file = tmp_path / "scores.jsonl"
    score_file.write_text("".join(
        json.dumps({"id": identifier, "scores": scores}, ensure_ascii=False) + "\n"
        for identifier in (1, 2)
    ), encoding="utf-8")
    tiers = tmp_path / "tiers.jsonl"
    tiers.write_text(
        json.dumps({"id": 1, "tier": "easy"}) + "\n" + json.dumps({"id": 2, "tier": "hard"}) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "ranking.json"
    sys.argv = ["apply", "--pairs", str(pairs), "--scores", str(score_file), "--output", str(output),
                "--replace-tiers", "hard", "--tiers", str(tiers)]
    apply_scores.main()
    ranking = json.loads(output.read_text(encoding="utf-8"))
    assert ranking["2"] == ["r|3", "r|2", "r|1"]
    assert ranking["1"] != ranking["2"]


def test_training_export_keeps_hard_negatives_and_drops_unreachable_gold(tmp_path):
    """Gold retrieval never surfaced has no text to train on, and must not be counted.

    Treating it as a positive would report a positive rate the training file does
    not contain, and would hide a candidate-generation gap behind a ranking one.
    """
    export = load_script("export_rerank_training")
    labels = tmp_path / "labels.jsonl"
    labels.write_text(json.dumps({
        "id": 1, "question": "q",
        "annotation": {
            "gold_tables": ["r|1", "r|99"], "gold_reports": ["r"],
            "row_column_bindings": [{"table": "r|1"}, {"table": "r|99"}],
        },
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text(json.dumps({
        "id": 1, "question": "q", "line_items": ["doanh thu"],
        "candidates": [
            {"table_id": "r|1", "sparse_rank": 1, "text": "gold"},
            {"table_id": "r|2", "sparse_rank": 2, "text": "hard negative"},
            {"table_id": "r|3", "sparse_rank": 3, "text": "easy negative"},
        ],
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    output = tmp_path / "training.jsonl"
    sys.argv = ["export", "--labels", str(labels), "--pairs", str(pairs),
                "--output", str(output), "--negatives", "1"]
    export.main()
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    # r|99 is gold but was never retrieved, so it contributes no row at all.
    assert [row["table_id"] for row in rows] == ["r|1", "r|2"]
    assert [row["label"] for row in rows] == [1, 0]
    # The negative kept is the best-ranked one, which is the comparison that decides.
    assert rows[1]["document"] == "hard negative"
    # The query carries the line item, exactly as the scoring notebook builds it.
    assert rows[0]["query"] == "q\nChỉ tiêu cần tìm: doanh thu"


def test_ranking_comparison_pairs_per_question_and_splits_by_tier():
    """The comparison has to see a reordering that helps and one that does not."""
    compare = load_script("compare_rankings")
    traces = [
        {"id": 1, "selected_docs": ["r"], "ranked_tables": ["r|1", "r|2"], "gold_tables_binding": ["r|2"]},
        {"id": 2, "selected_docs": ["r"], "ranked_tables": ["r|1", "r|2"], "gold_tables_binding": ["r|1"]},
    ]
    sparse = compare.scored(traces, None, 1, "binding")
    promoted = compare.scored(traces, {"1": ["r|2", "r|1"]}, 1, "binding")
    assert sparse[1]["recall"] == 0.0 and promoted[1]["recall"] == 1.0
    # Question 2 is untouched by that ranking, so the pairing isolates the change.
    assert promoted[2] == sparse[2]
    assert compare.means(promoted, [1])["f2"] == 1.0
