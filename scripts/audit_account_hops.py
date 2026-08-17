#!/usr/bin/env python3
"""What the account-code and note hops find, measured against the benchmark.

The hop is deterministic: the question names a line item, the lexicon says which
`Mã số` the corpus files that item under, the statements in the gated reports say
which of them carry that code, and each of those rows names the note that details
it. No model and no labels take part.

The benchmark's gold was bound from tables the pipeline already retrieves, so it
under-counts anything the hop finds that the ranker never proposed. Precision is
therefore the number to read here, and recall is a floor.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

from vifinqa.jsonl import load_jsonl
from vifinqa.lexicon import load_lexicon, resolve
from vifinqa.statements import normalize_label
from vifinqa.retrieval import table_budget

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, default=ROOT / "output/rerank/pairs_bench_v4.jsonl")
    parser.add_argument("--structure", type=Path, default=ROOT / "data/derived/table_structure.jsonl")
    parser.add_argument("--lexicon", type=Path, default=ROOT / "data/derived/account_lexicon.json")
    parser.add_argument("--cache", type=Path, default=ROOT / "output/diagnostics/traces.jsonl")
    parser.add_argument("--limit", type=int, default=2, help="codes kept per line item")
    args = parser.parse_args()

    labels = load_lexicon(args.lexicon)
    structure = list(load_jsonl(args.structure))
    by_report = defaultdict(list)
    for row in structure:
        by_report[row["report_id"]].append(row)
    note_at = {(row["report_id"], row["note_number"]): row["id"]
               for row in structure if row["note_number"] and row["kind"] == "note"}
    titled = defaultdict(set)
    for row in structure:
        if row["kind"] == "note" and row["heading"]:
            titled[row["report_id"]].add((row["heading"], row["id"]))

    pairs = {record["id"]: record for record in load_jsonl(args.pairs)}
    traces = [trace for trace in load_jsonl(args.cache) if trace["id"] in pairs]

    rows = []
    for trace in traces:
        record = pairs[trace["id"]]
        items = [normalize_label(item) for item in record.get("line_items") or []]
        codes = {code for item in items for code in resolve(item, labels, args.limit)}
        statements, notes, titles = set(), set(), set()
        for report in trace["selected_docs"]:
            for heading, identifier in titled.get(report, ()):
                if any(item == heading or item in heading for item in items):
                    titles.add(identifier)
        for report in trace["selected_docs"]:
            for table in by_report.get(report, ()):
                if table["kind"] != "statement" or not codes.intersection(table["codes"]):
                    continue
                statements.add(table["id"])
                for code in codes.intersection(table["note_by_code"]):
                    note = note_at.get((report, table["note_by_code"][code]))
                    if note:
                        notes.add(note)
        gold = set(trace["gold_tables_binding"])
        pool = set(record["candidates"] if isinstance(record["candidates"][0], str)
                   else [c["table_id"] for c in record["candidates"]])
        rows.append({
            "codes": len(codes),
            "statements": len(statements),
            "notes": len(notes),
            "hit_statements": len(statements & gold),
            "hit_notes": len(notes & gold),
            "titles": len(titles),
            "hit_titles": len(titles & gold),
            "gold": len(gold),
            "budget": table_budget(len(trace["selected_docs"]), "auto"),
            "outside_pool": len((statements | notes | titles) - pool),
            "gold_outside_pool": len(((statements | notes | titles) - pool) & gold),
        })

    def total(key):
        return sum(row[key] for row in rows)

    hop = total("statements") + total("notes")
    print(json.dumps({
        "questions": len(rows),
        "codes_per_question": round(statistics.mean(r["codes"] for r in rows), 2),
        "hop_tables_per_question": round(hop / len(rows), 2),
        "budget_per_question": round(statistics.mean(r["budget"] for r in rows), 2),
        "statement_precision": round(total("hit_statements") / max(total("statements"), 1), 4),
        "note_precision": round(total("hit_notes") / max(total("notes"), 1), 4),
        "titled_notes_per_question": round(total("titles") / len(rows), 2),
        "titled_note_precision": round(total("hit_titles") / max(total("titles"), 1), 4),
        "hop_precision": round((total("hit_statements") + total("hit_notes")) / max(hop, 1), 4),
        "hop_recall_of_binding_gold": round((total("hit_statements") + total("hit_notes")) / total("gold"), 4),
        "hop_tables_outside_the_pool": total("outside_pool"),
        "of_those_binding_gold": total("gold_outside_pool"),
    }, indent=2))


if __name__ == "__main__":
    main()
