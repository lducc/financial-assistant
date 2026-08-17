#!/usr/bin/env python3
"""Add the tables that literally carry the question's line item.

The ranker sees fifty candidates chosen by BM25 over one matched row, and its
budget then cuts to about five. A table that names the asked line item in its own
first column is evidence by construction, and the benchmark says we leave those
on the floor: adding them to the hard tier moves F2 from 0.6676 to 0.6965, CI
[+0.0166, +0.0422], with no other tier touched, because hard questions are the
ones whose gold spans several reports and several statements.

The index is built from the questions being answered and the corpus alone — a row
label matches when the item is the label or begins it, which is how questions
abbreviate ("lợi nhuận sau thuế" for "lợi nhuận sau thuế thu nhập doanh nghiệp").
No labels of ours take part, so the whole artefact regenerates from this script.

Only `relevant_tables` grows. Evidence, the CSVs and the answer path are
untouched, so nothing here can move execution accuracy.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from export_rerank_pairs import original_spans
from propose_multihop_labels import named_line_items
from vifinqa.jsonl import load_jsonl
from vifinqa.statements import normalize_label, rows


def question_items(question: str) -> set[str]:
    return {normalize_label(span) for span in original_spans(question, named_line_items(question))} - {""}


def index_corpus(catalog: Path, corpus: Path, wanted: set[str]) -> dict[str, set[str]]:
    """Table ids that carry each wanted label in a row's first column."""
    by_report = defaultdict(list)
    for record in load_jsonl(catalog):
        by_report[record["source_path"]].append(record)
    found = defaultdict(set)
    for source, tables in by_report.items():
        lines = (corpus / source).read_text(encoding="utf-8", errors="replace").split("\n")
        for record in tables:
            if record["start_line"] > len(lines):
                continue
            for row in rows(lines[record["start_line"] - 1]):
                if not row:
                    continue
                words = normalize_label(row[0]).split()
                for size in range(1, min(len(words), 14) + 1):
                    prefix = " ".join(words[:size])
                    if prefix in wanted:
                        found[prefix].add(record["submission_table_id"])
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, default=ROOT / "data/raw/vifinqa/questions/questions.jsonl")
    parser.add_argument("--corpus", type=Path, default=ROOT / "data/raw/vifinqa")
    parser.add_argument("--catalog", type=Path, default=ROOT / "data/derived/table_catalog/tables.jsonl")
    parser.add_argument("--ranking", type=Path, required=True, help="the ranking that names each question's gated reports")
    parser.add_argument("--tiers", type=Path, default=ROOT / "data/derived/question_tiers.jsonl")
    parser.add_argument("--tier", default="hard", help="comma-separated tiers to expand, or 'all'")
    parser.add_argument("--output", type=Path, default=ROOT / "output/rerank/expansion.json")
    args = parser.parse_args()

    questions = list(load_jsonl(args.questions))
    items = {question["id"]: question_items(question["question"]) for question in questions}
    index = index_corpus(args.catalog, args.corpus, set().union(*items.values()))

    ranking = json.loads(args.ranking.read_text(encoding="utf-8"))
    tier_of = {record["id"]: record["tier"] for record in load_jsonl(args.tiers)}
    gate = {tier.strip() for tier in args.tier.split(",")} if args.tier != "all" else set(tier_of.values())

    expansion = {}
    for question in questions:
        identifier = question["id"]
        if tier_of.get(identifier) not in gate:
            continue
        reports = {table.split("|")[0] for table in ranking.get(str(identifier), [])}
        tables = {table for item in items[identifier] for table in index.get(item, ())
                  if table.split("|")[0] in reports}
        if tables:
            expansion[str(identifier)] = sorted(tables)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(expansion, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({
        "questions": len(questions),
        "expanded": len(expansion),
        "tables_added": sum(len(tables) for tables in expansion.values()),
        "labels_indexed": len(index),
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
