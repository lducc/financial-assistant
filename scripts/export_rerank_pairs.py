#!/usr/bin/env python3
"""Export question/table candidate pairs for cross-encoder reranking off-box.

Reranking is the one stage the teams above us all run and we do not. A
cross-encoder scores a question against a table representation, so the pairs have
to be materialized where the ranking is known and scored where a GPU is: this
writes them, a notebook scores them, and apply_rerank_scores.py fuses the result.

Only the ordering of already-retrieved candidates changes. The document gate, the
candidate set, and the table IDs are untouched, so a bad reranker can be dropped
by ignoring the scores file.
"""

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from docs import load_companies, load_reports as load_doc_reports, parse_question, required_report_years, retrieve_docs
from vifinqa.rerank import table_representation
from vifinqa.retrieval import load_reports, retrieve_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data" / "raw" / "vifinqa")
    parser.add_argument("--questions", type=Path, help="defaults to the full question set")
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "rerank" / "pairs.jsonl")
    parser.add_argument("--depth", type=int, default=50, help="candidates per question to rerank")
    parser.add_argument("--table-mode", default="report-coverage")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--inventory", type=int, default=600, help="characters of the table's line-item list to include; 0 keeps the matched row alone")
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()

    path = args.questions or args.dataset_root / "questions" / "questions.jsonl"
    questions = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        questions = questions[: args.limit]

    companies = load_companies(args.dataset_root / "code_stock.csv")
    doc_reports = load_doc_reports(args.dataset_root / "financial_statements")
    table_reports = load_reports(args.dataset_root)
    by_id = {report.identity.report_id: report for report in table_reports}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pairs = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for number, question in enumerate(questions, 1):
            parsed = parse_question(question["question"], companies)
            if not parsed.tickers and parsed.candidate_tickers:
                parsed.tickers = parsed.candidate_tickers[:1]
            docs, _ = retrieve_docs(parsed, doc_reports)
            result = retrieve_rows(
                question["question"],
                {
                    "tickers": parsed.tickers,
                    "years": parsed.years,
                    "slot_years": required_report_years(parsed),
                    "scope": parsed.scope,
                },
                table_reports,
                top_k=args.depth,
                report_ids=docs,
                mode=args.table_mode,
            )
            candidates = []
            for rank, table in enumerate(result["tables"], 1):
                report = by_id[table["report_id"]]
                match = next(
                    (item for item in report_tables_cached(report) if item.table_id == table["table_id"]), None
                )
                if match is None:
                    continue
                candidates.append({
                    "table_id": table["table_id"],
                    "sparse_rank": rank,
                    "text": table_representation(match, table["row_index"], inventory=args.inventory),
                })
            handle.write(json.dumps({
                "id": question["id"],
                "question": question["question"],
                "selected_docs": docs,
                "candidates": candidates,
            }, ensure_ascii=False) + "\n")
            pairs += len(candidates)
            if args.progress_every and number % args.progress_every == 0:
                print(f"exported {number}/{len(questions)} questions, {pairs} pairs", flush=True, file=sys.stderr)

    print(json.dumps({
        "output": str(args.output),
        "questions": len(questions),
        "pairs": pairs,
        "megabytes": round(args.output.stat().st_size / 1e6, 1),
    }, ensure_ascii=False, indent=2))


def report_tables_cached(report):
    from vifinqa.retrieval import report_tables
    return report_tables(str(report.path), report.identity)


if __name__ == "__main__":
    main()
