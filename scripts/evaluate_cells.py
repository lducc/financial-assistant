#!/usr/bin/env python3
"""Score cell selection against the benchmark's verified row/column bindings.

Retrieval metrics stop at the table. Execution accuracy depends on the cell, and
the benchmark already pins every figure down to (table, row, column, raw string) —
613 bindings verified against raw OCR. That is a local oracle for the answer path,
needing none of the organizer's answers.

Each gold binding is attributed to the first stage that lost it:

    table_missed   the table never reached the submission
    row_missed     right table, retrieval bound a different row
    column_missed  right row, the value column chosen was the wrong period
    hit            table, row and column all correct

Column errors are the interesting ones: a balance sheet carries "Số cuối năm" and
"Số đầu năm" side by side, so picking the wrong one returns a real figure from the
wrong period, which no downstream arithmetic can recover from.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]

from docs import load_companies, load_reports as load_doc_reports, parse_question, required_report_years, retrieve_docs
from vifinqa.answers import first_numeric_cell, select_cell
from vifinqa.jsonl import load_jsonl
from vifinqa.retrieval import load_reports, metric_query_tokens, report_tables, retrieve_rows, table_budget


def selections(result: dict, question: str, reports: dict, search: bool) -> dict[str, tuple[int, int | None]]:
    """What the answer path would read from each submitted table."""
    chosen = {}
    tokens = {token for token in metric_query_tokens(question, {"tickers": [], "years": []}) if len(token) > 2}
    for table in result["tables"]:
        if search:
            report = reports[table["report_id"]]
            parsed = next(
                (item for item in report_tables(str(report.path), report.identity)
                 if item.table_id == table["table_id"]), None,
            )
            picked = select_cell([list(row) for row in parsed.rows], tokens, table.get("header_cells")) if parsed else None
            if picked is not None:
                chosen[table["table_id"]] = (picked[0], picked[1])
                continue
        numeric = first_numeric_cell(table["row_cells"], table.get("header_cells"))
        chosen[table["table_id"]] = (table["row_index"], numeric[0] if numeric else None)
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data" / "raw" / "vifinqa")
    parser.add_argument("--benchmark", type=Path, default=ROOT / "annotations" / "benchmark.jsonl")
    parser.add_argument("--table-mode", default="report-coverage")
    parser.add_argument("--table-top-k", default="auto")
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "cells" / "report.json")
    parser.add_argument("--search-rows", action="store_true", help="search every row of a submitted table instead of trusting the retrieval binding")
    parser.add_argument("--progress-every", type=int, default=50)
    args = parser.parse_args()

    records = load_jsonl(args.benchmark)
    companies = load_companies(args.dataset_root / "code_stock.csv")
    doc_reports = load_doc_reports(args.dataset_root / "financial_statements")
    table_reports = load_reports(args.dataset_root)
    reports_by_id = {report.identity.report_id: report for report in table_reports}

    outcomes: Counter = Counter()
    by_tier: dict[str, Counter] = {}
    misses = []
    for number, record in enumerate(records, 1):
        parsed = parse_question(record["question"], companies)
        if not parsed.tickers and parsed.candidate_tickers:
            parsed.tickers = parsed.candidate_tickers[:1]
        docs, _ = retrieve_docs(parsed, doc_reports)
        result = retrieve_rows(
            record["question"],
            {
                "tickers": parsed.tickers,
                "years": parsed.years,
                "slot_years": required_report_years(parsed),
                "scope": parsed.scope,
            },
            table_reports,
            top_k=table_budget(len(docs), args.table_top_k),
            report_ids=docs,
            mode=args.table_mode,
        )
        chosen = selections(result, record['question'], reports_by_id, args.search_rows)
        tier = by_tier.setdefault(record["tier"], Counter())
        for binding in record["annotation"]["row_column_bindings"]:
            if binding["table"] not in chosen:
                outcome = "table_missed"
            else:
                row, column = chosen[binding["table"]]
                if row != binding["row"]:
                    outcome = "row_missed"
                elif column != binding["column"]:
                    outcome = "column_missed"
                else:
                    outcome = "hit"
            outcomes[outcome] += 1
            tier[outcome] += 1
            if outcome in {"row_missed", "column_missed"}:
                misses.append({
                    "id": record["id"],
                    "outcome": outcome,
                    "table": binding["table"],
                    "gold_row": binding["row"],
                    "gold_column": binding["column"],
                    "gold_raw": binding["raw"],
                    "gold_label": binding["row_label"][:60],
                    "chosen_row": chosen[binding["table"]][0],
                    "chosen_column": chosen[binding["table"]][1],
                })
        if args.progress_every and number % args.progress_every == 0:
            print(f"scored {number}/{len(records)}", flush=True, file=sys.stderr)

    total = sum(outcomes.values())
    report = {
        "bindings": total,
        "outcomes": {name: outcomes[name] for name in ("hit", "column_missed", "row_missed", "table_missed")},
        "share": {
            name: round(outcomes[name] / total, 4)
            for name in ("hit", "column_missed", "row_missed", "table_missed")
        },
        "by_tier": {
            tier: {
                "bindings": sum(counts.values()),
                **{name: round(counts[name] / sum(counts.values()), 3) for name in ("hit", "column_missed", "row_missed", "table_missed")},
            }
            for tier, counts in sorted(by_tier.items())
        },
        "example_misses": misses[:15],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("bindings", "outcomes", "share", "by_tier")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
