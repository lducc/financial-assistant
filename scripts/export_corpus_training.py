#!/usr/bin/env python3
"""Build reranker training data out of the corpus, using none of our labels.

The pool holds 87.4% of the gold live and the shipped order surfaces 61.5% of it,
so 0.22 F2 sits in the ranking stage and nothing else is worth chasing. Every
zero-shot lever on that stage is spent, and every hand-written promotion rule
loses to the model: coverage ordering -0.008, account-code promotion -0.050,
listwise generation -0.046. What is left is a better scorer.

Fine-tuning was blocked once on data — 312 labelled questions, four of them hard,
too thin for LoRA to beat a strong 8B. The corpus removes that. A table's own row
labels are the questions it answers, so a query synthesized from a label has that
table as a positive and the rest of its report as negatives. This is Doc2Query and
InPars applied to filings: no gold, nothing to overfit to the leaderboard, and
167,306 label observations to draw on instead of 1,130 positives.

Two things keep it honest. Reports the benchmark touches are excluded, so the 233
questions stay a held-out set. And negatives come only from the positive's own
report, which is the discrimination that actually decides our ranking — the
document gate is 0.97 precise, so the model is never asked to tell one company
from another, only one line item from its neighbour.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from docs import load_companies
from vifinqa.jsonl import load_jsonl, write_jsonl
from vifinqa.rerank import table_representation
from vifinqa.retrieval import load_reports, report_tables
from vifinqa.statements import normalize_label

SCOPE = {"separate": "công ty mẹ"}


def company_names(path: Path) -> dict[str, str]:
    loaded = load_companies(path)
    entries = loaded.values() if isinstance(loaded, dict) else loaded
    return {entry.ticker: getattr(entry, "name", entry.ticker) for entry in entries}


def question_for(label: str, identity, name: str) -> str:
    scope = SCOPE.get(identity.scope, "")
    who = f"{scope} {name} ({identity.ticker})".strip()
    return f"{label} năm {identity.year} của {who} là bao nhiêu?"


def rows_for(report, companies, per_report, negatives, rng):
    tables = report_tables(str(report.path), report.identity)
    if len(tables) < negatives + 1:
        return []
    holders = defaultdict(list)
    original = {}
    for index, table in enumerate(tables):
        for row_index, row in enumerate(table.rows):
            label = normalize_label(row[0]) if row and row[0] else ""
            if not 8 <= len(label) <= 80 or len(label.split()) > 12:
                continue
            holders[label].append((index, row_index))
            original.setdefault(label, row[0].strip())
    # A label in one or two tables identifies them; one in twenty is boilerplate
    # and teaches the model nothing about which table answers the question.
    usable = [label for label, seen in holders.items() if 1 <= len(seen) <= 2]
    if not usable:
        return []
    name = companies.get(report.identity.ticker, report.identity.ticker)
    rows = []
    for label in rng.sample(usable, min(per_report, len(usable))):
        positives = holders[label]
        chosen = {index for index, _ in positives}
        pool = [index for index in range(len(tables)) if index not in chosen]
        query = f"{question_for(original[label], report.identity, name)}\nChỉ tiêu cần tìm: {original[label]}"
        identifier = f"{report.identity.report_id}:{label}"
        for index, row_index in positives:
            rows.append({
                "id": identifier, "query": query, "label": 1,
                "table_id": f"{report.identity.report_id}|{tables[index].start_line}",
                "document": table_representation(tables[index], row_index, inventory=600),
            })
        for index in rng.sample(pool, min(negatives, len(pool))):
            rows.append({
                "id": identifier, "query": query, "label": 0,
                "table_id": f"{report.identity.report_id}|{tables[index].start_line}",
                "document": table_representation(tables[index], 0, inventory=600),
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/raw/vifinqa")
    parser.add_argument("--cache", type=Path, default=ROOT / "output/diagnostics/traces.jsonl")
    parser.add_argument("--benchmark", type=Path, default=ROOT / "annotations/benchmark.jsonl")
    parser.add_argument("--per-report", type=int, default=4)
    parser.add_argument("--negatives", type=int, default=6)
    parser.add_argument("--reports", type=int, help="stop after this many reports")
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--output", type=Path, default=ROOT / "output/rerank/training_corpus.jsonl")
    args = parser.parse_args()

    held_out = {r["id"] for r in load_jsonl(args.benchmark)}
    excluded = {report for trace in load_jsonl(args.cache) if trace["id"] in held_out
                for report in trace["selected_docs"]}
    companies = company_names(args.data_root / "code_stock.csv")

    rng = random.Random(args.seed)
    reports = [r for r in load_reports(args.data_root) if r.identity.report_id not in excluded]
    rng.shuffle(reports)
    if args.reports:
        reports = reports[:args.reports]

    rows = []
    for number, report in enumerate(reports, 1):
        rows.extend(rows_for(report, companies, args.per_report, args.negatives, rng))
        if number % 200 == 0:
            print(f"{number}/{len(reports)} reports, {len(rows)} rows", flush=True)

    write_jsonl(args.output, rows)
    groups = len({row["id"] for row in rows})
    print(json.dumps({
        "reports_used": len(reports), "reports_excluded": len(excluded),
        "groups": groups, "rows": len(rows),
        "positives": sum(row["label"] for row in rows),
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
