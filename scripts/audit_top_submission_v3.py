#!/usr/bin/env python3
"""Robust, read-only audit of the public ViFinQA leader artifact."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
import json
from pathlib import Path
import re


# Accept both ``candidate_id == 'x'`` and Pandas ``df['candidate_id'] == 'x'``.
QUERY_ID = re.compile(r"candidate_id['\"\]]*\s*==\s*['\"]([^'\"]+)['\"]")


def read_rows(root: Path, record_id: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted((root / "data").glob(f"q{record_id:04d}_df*.csv")):
        with path.open(encoding="utf-8", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission_root", type=Path)
    args = parser.parse_args()
    records = json.loads((args.submission_root / "submission.json").read_text(encoding="utf-8"))
    row_counts: list[int] = []
    query_hits = answer_hits = missing_answer_value = no_candidate = no_csv = 0
    line_table_deltas: list[int] = []
    precomputed_columns = Counter()
    examples: list[dict[str, object]] = []
    for record in records:
        rows = read_rows(args.submission_root, int(record["id"]))
        if not rows:
            no_csv += 1
            continue
        row_counts.append(len(rows))
        for row in rows:
            precomputed_columns.update(c for c in row if c in {"answer_value", "retrieval_score", "raw_number"})
        match = QUERY_ID.search(record.get("pandas_query", ""))
        candidate_id = match.group(1) if match else None
        target = next((row for row in rows if candidate_id and row.get("candidate_id") == candidate_id), None)
        if target is None:
            no_candidate += 1
            if len(examples) < 5:
                examples.append({"id": record["id"], "kind": "candidate_not_found", "query": record.get("pandas_query", "")})
            continue
        query_hits += 1
        raw_answer = target.get("answer_value", "")
        if not raw_answer:
            missing_answer_value += 1
        else:
            try:
                if abs(float(raw_answer) - float(record["answer"])) <= 1e-9:
                    answer_hits += 1
            except (TypeError, ValueError):
                missing_answer_value += 1
        if record.get("relevant_tables") and target.get("table_id", "").strip():
            try:
                submitted_line = int(str(record["relevant_tables"][0]).rsplit("|", 1)[1])
                csv_table_id = int(float(target["table_id"]))
            except (IndexError, ValueError):
                continue
            delta = submitted_line - csv_table_id
            line_table_deltas.append(delta)
            if len(examples) < 5:
                examples.append({"id": record["id"], "kind": "matched_candidate", "submitted_table": record["relevant_tables"][0], "csv_table_id": target["table_id"], "delta": delta, "answer": record["answer"], "answer_value": raw_answer})
    print(json.dumps({
        "records": len(records),
        "csv_row_count_distribution": dict(Counter(row_counts)),
        "questions_without_csv": no_csv,
        "pandas_query_candidate_id_hits": query_hits,
        "questions_without_matching_candidate": no_candidate,
        "answer_equals_selected_answer_value": answer_hits,
        "selected_candidates_missing_answer_value": missing_answer_value,
        "precomputed_columns": dict(precomputed_columns),
        "submitted_line_minus_csv_table_id": dict(Counter(line_table_deltas)),
        "examples": examples,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
