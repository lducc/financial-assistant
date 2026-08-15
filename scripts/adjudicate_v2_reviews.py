#!/usr/bin/env python3
"""Create a non-destructive, auditable v2 adjudication from two reviews."""

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

from docs import load_companies, load_reports as load_document_reports, parse_question, retrieve_docs
from vifinqa.evaluation_v2 import index_records, load_jsonl, validate_v2_source_bindings
from vifinqa.retrieval import load_reports as load_table_reports, report_tables, retrieve_rows, tokenize


def review_reports(record: dict) -> set[str]:
    return {
        report
        for slot in record["slots"]
        for alternative in slot["alternatives"]
        for report in alternative["reports"]
    }


def gated_reports(question: str, companies: dict, reports: dict) -> set[str]:
    parsed = parse_question(question, companies)
    if not parsed.tickers and parsed.candidate_tickers:
        parsed.tickers = parsed.candidate_tickers[:1]
    selected, _ = retrieve_docs(parsed, reports)
    return set(selected)


def best_row(table: dict, question: str) -> tuple[int, int, str, str]:
    """Choose a source-bound evidence cell using lexical overlap with the question."""
    question_tokens = set(tokenize(question))
    rows = table["rows"]
    row_index, row = max(
        enumerate(rows),
        key=lambda item: (len(question_tokens & set(tokenize(" ".join(item[1])))), -item[0]),
    )
    column = next((index for index, value in enumerate(row[1:], 1) if any(char.isdigit() for char in value)), 0)
    return row_index, column, row[column], row[0] if row else question


def lexical_resolution(
    question: str, expected_reports: set[str], reviewer_records: list[dict], reports: list,
) -> list[dict]:
    """Resolve one table per gated report from reviewer evidence or OCR retrieval."""
    reviewer_tables = {
        table
        for record in reviewer_records
        for slot in record["slots"]
        for alternative in slot["alternatives"]
        for table in alternative["tables"]
        if table.partition("|")[0] in expected_reports
    }
    retrieved = retrieve_rows(question, {}, reports, report_ids=sorted(expected_reports), mode="report-coverage")
    table_by_id = {
        table.table_id: table
        for report in reports
        if report.identity.report_id in expected_reports
        for table in report_tables(str(report.path), report.identity)
    }
    report_by_id = {report.identity.report_id: report for report in reports}
    selected = []
    for report_id in sorted(expected_reports):
        candidates = [table_id for table_id in reviewer_tables if table_id.partition("|")[0] == report_id]
        candidates.extend(
            table["table_id"] for table in retrieved["tables"] if table["report_id"] == report_id
        )
        valid = [table_by_id[table_id] for table_id in dict.fromkeys(candidates) if table_id in table_by_id]
        if not valid:
            continue
        question_tokens = set(tokenize(question))
        table = max(
            valid,
            key=lambda item: max((len(question_tokens & set(tokenize(" ".join(row)))) for row in item.rows), default=0),
        )
        row, column, raw, metric = best_row({"rows": table.rows}, question)
        selected.append({
            "slot_id": f"{report_id}|{table.start_line}|value", "entity": report_id.split("_", 1)[0],
            "report_year": report_by_id[report_id].identity.year,
            "scope": report_by_id[report_id].identity.scope,
            "metric": metric or question, "operand_role": "value",
            "alternatives": [{"reports": [report_id], "tables": [table.table_id], "cells": [{
                "table": table.table_id, "row": row, "column": column, "raw": raw,
                "period": "unknown", "unit": "unknown",
            }]}],
        })
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviewer-a", type=Path, required=True)
    parser.add_argument("--reviewer-b", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument("--resolve-lexically", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite adjudication: {args.output}")

    reviews = {
        "reviewer_a": index_records(load_jsonl(args.reviewer_a), "reviewer A"),
        "reviewer_b": index_records(load_jsonl(args.reviewer_b), "reviewer B"),
    }
    if set(reviews["reviewer_a"]) != set(reviews["reviewer_b"]):
        raise ValueError("reviewer IDs must match exactly")
    queue = index_records(load_jsonl(args.queue), "queue")
    if set(queue) != set(reviews["reviewer_a"]):
        raise ValueError("queue and reviewer IDs must match exactly")
    companies = load_companies(args.raw_root / "code_stock.csv")
    documents = load_document_reports(args.raw_root / "financial_statements")
    table_reports = load_table_reports(args.raw_root) if args.resolve_lexically else []
    records = []
    unresolved = []
    for identifier in sorted(queue):
        for reviewer, records_by_id in reviews.items():
            errors = validate_v2_source_bindings(records_by_id[identifier], args.raw_root)
            if errors:
                raise ValueError(f"{reviewer} id={identifier} source errors: {errors[:3]}")
        expected_reports = gated_reports(queue[identifier]["question"], companies, documents)
        scores = {
            reviewer: len(review_reports(records_by_id[identifier]) & expected_reports)
            for reviewer, records_by_id in reviews.items()
        }
        best_score = max(scores.values())
        selected_names = [name for name, score in scores.items() if score == best_score and score > 0]
        if len(selected_names) != 1:
            if args.resolve_lexically:
                slots = lexical_resolution(
                    queue[identifier]["question"], expected_reports,
                    [reviews["reviewer_a"][identifier], reviews["reviewer_b"][identifier]], table_reports,
                )
                if slots:
                    selected = {key: value for key, value in reviews["reviewer_a"][identifier].items()}
                    selected["operation"] = "lookup"
                    selected["confidence"] = 0.5
                    selected["slots"] = slots
                    selected_key = "raw_ocr_lexical_resolution"
                else:
                    unresolved.append({"id": identifier, "expected_reports": sorted(expected_reports), "scores": scores})
                    continue
            else:
                unresolved.append({
                    "id": identifier,
                    "question": queue[identifier]["question"],
                    "expected_reports": sorted(expected_reports),
                    "scores": scores,
                    "reviewer_a": reviews["reviewer_a"][identifier],
                    "reviewer_b": reviews["reviewer_b"][identifier],
                })
                continue
        else:
            selected_key = selected_names[0]
            selected = reviews[selected_key][identifier]
        resolved = {key: value for key, value in selected.items()}
        resolved["adjudication"] = {
            "resolved": True,
            "selected_review": selected_key,
            "reason": "selected review has the strongest overlap with the question's document-gated reports" if selected_key != "raw_ocr_lexical_resolution" else "raw OCR lexical resolution within document-gated report(s)",
            "expected_reports": sorted(expected_reports),
            "report_overlap": scores,
            "reviewer_alternatives": {
                reviewer: records_by_id[identifier]["slots"]
                for reviewer, records_by_id in reviews.items()
            },
        }
        records.append(resolved)
    if unresolved:
        if args.audit_output:
            if args.audit_output.exists():
                raise FileExistsError(f"refusing to overwrite adjudication audit: {args.audit_output}")
            args.audit_output.parent.mkdir(parents=True, exist_ok=True)
            args.audit_output.write_text(
                "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in unresolved),
                encoding="utf-8",
            )
        preview = ", ".join(str(item["id"]) for item in unresolved[:10])
        raise ValueError(f"unresolved adjudications={len(unresolved)} ids={preview}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
