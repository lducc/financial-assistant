#!/usr/bin/env python3
"""Rebuild retrieval labels from preserved Gold-150 source evidence.

This restores document/table/cell ground truth only. Original answer programs,
operation taxonomy, and numeric-answer assertions are not recoverable from the
preserved evidence CSV and are intentionally not invented here.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, default=ROOT / "data" / "results" / "e027_gold_150" / "evidence.csv")
    parser.add_argument("--questions", type=Path, default=ROOT / "data" / "raw" / "vifinqa" / "questions" / "questions.jsonl")
    parser.add_argument(
        "--candidate-submission", type=Path,
        default=ROOT / "data" / "results" / "top_synera_2333" / "submission_unpacked" / "submission.json",
    )
    parser.add_argument(
        "--candidate-output", type=Path,
        default=ROOT / "annotations" / "gold_150_candidate_submissions.jsonl",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "annotations" / "gold_150.jsonl")
    args = parser.parse_args()

    questions = {
        int(row["id"]): row["question"]
        for line in args.questions.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for row in [json.loads(line)]
    }
    candidates = {}
    if args.candidate_submission.is_file():
        candidates = {
            int(row["id"]): row
            for row in json.loads(args.candidate_submission.read_text(encoding="utf-8"))
        }
    bindings: dict[int, list[dict]] = defaultdict(list)
    with args.evidence.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            identifier = int(row["question_id"])
            report_id = row["report_id"]
            bindings[identifier].append({
                "role": row["role"],
                "table": row["table"],
                "row": int(row["row"]),
                "column": int(row["column"]),
                "row_label": row["row_label"],
                "column_header": row["column_header"],
                "raw": row["raw"],
                "source_path": f"financial_statements/{row['ticker']}/{row['year']}/{report_id}/{report_id}_extracted.txt",
            })

    records = []
    candidate_records = []
    for identifier in sorted(bindings):
        cells = bindings[identifier]
        tables = list(dict.fromkeys(cell["table"] for cell in cells))
        reports = list(dict.fromkeys(table.partition("|")[0] for table in tables))
        records.append({
            "id": identifier,
            "question": questions[identifier],
            "taxonomy": {"operation": "unrecovered", "table_count": len(tables)},
            "annotation": {
                "status": "complete",
                "gold_reports": reports,
                "gold_tables": tables,
                "row_column_bindings": cells,
            },
            "provenance": {
                "reconstructed_from": "data/results/e027_gold_150/evidence.csv",
                "scope": "retrieval evidence only; original answer programs and taxonomy unavailable",
            },
        })
        if identifier in candidates:
            candidate_records.append({
                "id": identifier,
                "question": questions[identifier],
                "candidate_submission": {
                    key: candidates[identifier].get(key)
                    for key in ("answer", "relevant_docs", "relevant_tables", "evidence", "pandas_query")
                },
                "provenance": "unreviewed candidate submission; not gold",
            })
    if len(records) != 150:
        raise ValueError(f"expected 150 records, got {len(records)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    if candidate_records:
        args.candidate_output.write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in candidate_records),
            encoding="utf-8",
        )
    print(args.output)


if __name__ == "__main__":
    main()
