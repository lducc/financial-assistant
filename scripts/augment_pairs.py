#!/usr/bin/env python3
"""Let the tables that name the line item compete for the top of the ranking.

They are currently appended below the budget, unscored, which is why the ones we
submit are gold at 0.055 live against a 0.105 break-even — the ranker never sees
them, so they land wherever the tail happens to be. On the benchmark, where the
same tables are ranked rather than appended, they are gold at 0.370.

So they belong in the candidate set, not after it. Each arrives with the row the
question asks about as its matched row, which is the representation the ranker
reads, and with a sparse rank continuing past the pool so nothing already there
moves. Everything else about the export is untouched.
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_item_expansion import index_corpus, question_items
from vifinqa.jsonl import load_jsonl, write_jsonl
from vifinqa.lexicon import item_row
from vifinqa.rerank import table_representation
from vifinqa.retrieval import load_reports, report_tables


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/raw/vifinqa")
    parser.add_argument("--catalog", type=Path, default=ROOT / "data/derived/table_catalog/tables.jsonl")
    parser.add_argument("--inventory", type=int, default=600)
    args = parser.parse_args()

    records = list(load_jsonl(args.pairs))
    items = {record["id"]: sorted(question_items(record["question"])) for record in records}
    carriers = index_corpus(args.catalog, args.data_root, set().union(*items.values()))
    reports = {report.identity.report_id: report for report in load_reports(args.data_root)}

    tables_of, added = {}, 0
    for record in records:
        known = {candidate["table_id"] for candidate in record["candidates"]}
        rank = max((c["sparse_rank"] for c in record["candidates"]), default=0)
        wanted = {table for item in items[record["id"]] for table in carriers.get(item, ())
                  if table.split("|")[0] in record["selected_docs"] and table not in known}
        for table_id in sorted(wanted):
            report_id, _, line = table_id.partition("|")
            if report_id not in tables_of:
                report = reports[report_id]
                tables_of[report_id] = {t.start_line: t for t in report_tables(str(report.path), report.identity)}
            table = tables_of[report_id].get(int(line))
            if table is None:
                continue
            row = item_row([list(r) for r in table.rows], items[record["id"]]) or 0
            rank += 1
            added += 1
            record["candidates"].append({
                "table_id": table_id, "sparse_rank": rank,
                "text": table_representation(table, row, inventory=args.inventory),
            })

    write_jsonl(args.output, records)
    print(json.dumps({
        "questions": len(records),
        "candidates_added": added,
        "added_per_question": round(added / len(records), 2),
        "pairs_now": sum(len(r["candidates"]) for r in records),
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
