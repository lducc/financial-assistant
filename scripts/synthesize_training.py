#!/usr/bin/env python3
"""Synthesize reranker training data whose supervision is not the rule being learned.

The first attempt labelled a table positive when it wrote the question's line item
as a row, which is the lexical rule already measured at +0.0045 live — training on
it teaches the model to be a worse version of a grep. Two thirds of its negatives
shared no content word with the item, so they were rejectable without reading.

This builds the set three ways instead.

**Positives come from the figure, not the label.** A filing is internally
redundant: the balance-sheet row, the note that details it, the segment breakdown
and next year's comparative all print the same number. So a positive is any table
in the report carrying the same value, which finds the note whose wording differs
from the statement's — exactly the gold the ranker misses — without a label ever
being compared. Values need seven significant digits and must not be spread over
more than a few tables, or a repeated round number links everything to everything.

**Negatives come from the retrieval distribution.** The reranker sees BM25's
candidates at inference, so the negatives are BM25's top tables in that report
that do not carry the figure. Its mistakes are the sibling rows — "dự phòng giảm
giá hàng tồn kho" against "hàng tồn kho" — which is the discrimination the task
actually turns on.

**Queries come from the questions, not from a template I invented.** The 1,012
real questions are slotted on their line item, year and company and sampled back,
so phrasing, unit clauses and period wording match the distribution the model will
be scored against.

Reports the benchmark touches are excluded, so the 233 held-out questions stay
honest.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from docs import load_companies
from export_rerank_pairs import original_spans
from propose_multihop_labels import named_line_items
from vifinqa.jsonl import load_jsonl, write_jsonl
from vifinqa.rerank import table_representation
from vifinqa.retrieval import load_reports, report_tables, unicode_bm25, unicode_tokenize
from vifinqa.statements import CODE, normalize_label

DIGITS = re.compile(r"\d")
YEAR = re.compile(r"\b(19|20)\d{2}\b")


def figure(cell: str) -> str:
    """The digits of an amount, or empty if the cell is not one worth linking."""
    digits = "".join(DIGITS.findall(cell))
    return digits if len(digits.lstrip("0")) >= 7 and not cell.strip().endswith("%") else ""


def templates(questions: list[dict], companies: dict[str, str]) -> list[str]:
    """Real questions with their item, year and company slotted out."""
    found = []
    for question in questions:
        text = question["question"]
        items = original_spans(text, named_line_items(text))
        if len(items) != 1 or items[0] not in text:
            continue
        slotted = text.replace(items[0], "{item}", 1)
        slotted = YEAR.sub("{year}", slotted)
        for ticker, name in companies.items():
            if name and name in slotted:
                slotted = slotted.replace(name, "{company}").replace(ticker, "{ticker}")
                break
        else:
            continue
        if "{item}" in slotted and "{company}" in slotted:
            found.append(slotted)
    return found


def groups_for(report, shapes, companies, negatives, rng):
    tables = report_tables(str(report.path), report.identity)
    if len(tables) < negatives + 2:
        return []
    texts = [table_representation(table, 0, inventory=600) for table in tables]
    carries = defaultdict(set)
    for index, table in enumerate(tables):
        for row in table.rows:
            for cell in row:
                value = figure(cell)
                if value:
                    carries[value].add(index)

    anchors = []
    for index, table in enumerate(tables):
        for row_index, row in enumerate(table.rows):
            if row_index == 0 or not row or not row[0].strip():
                continue
            if not any(CODE.match(cell.strip()) for cell in row[1:3]):
                continue
            for cell in row[1:]:
                value = figure(cell)
                # One figure in two to four tables is a statement and the places
                # that restate it; in more it is a repeated round number.
                if value and 2 <= len(carries[value]) <= 4:
                    anchors.append((normalize_label(row[0]), row[0].strip(), value, index, row_index))
                    break

    rng.shuffle(anchors)
    seen, rows = set(), []
    name = companies.get(report.identity.ticker, report.identity.ticker)
    for label, original, value, index, row_index in anchors:
        if label in seen:
            continue
        seen.add(label)
        positives = carries[value]
        query = rng.choice(shapes).format(
            item=original, year=report.identity.year, company=name, ticker=report.identity.ticker,
        )
        ranked = unicode_bm25(unicode_tokenize(original), texts)
        hard = [i for i in sorted(range(len(texts)), key=lambda i: -ranked[i]) if i not in positives][:negatives]
        if len(hard) < negatives:
            continue
        identifier = f"{report.identity.report_id}:{value}"
        for i in sorted(positives):
            rows.append({
                "id": identifier, "query": query, "label": 1,
                "table_id": f"{report.identity.report_id}|{tables[i].start_line}",
                "document": table_representation(tables[i], row_index if i == index else 0, inventory=600),
            })
        for i in hard:
            rows.append({
                "id": identifier, "query": query, "label": 0,
                "table_id": f"{report.identity.report_id}|{tables[i].start_line}",
                "document": texts[i],
            })
        if len(seen) >= 8:
            break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/raw/vifinqa")
    parser.add_argument("--cache", type=Path, default=ROOT / "output/diagnostics/traces.jsonl")
    parser.add_argument("--benchmark", type=Path, default=ROOT / "annotations/benchmark.jsonl")
    parser.add_argument("--negatives", type=int, default=8)
    parser.add_argument("--reports", type=int)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--output", type=Path, default=ROOT / "output/rerank/training_linked.jsonl")
    args = parser.parse_args()

    held_out = {record["id"] for record in load_jsonl(args.benchmark)}
    excluded = {report for trace in load_jsonl(args.cache) if trace["id"] in held_out
                for report in trace["selected_docs"]}
    loaded = load_companies(args.data_root / "code_stock.csv")
    entries = loaded.values() if isinstance(loaded, dict) else loaded
    companies = {entry.ticker: getattr(entry, "name", entry.ticker) for entry in entries}

    questions = list(load_jsonl(args.data_root / "questions" / "questions.jsonl"))
    shapes = templates(questions, companies)
    if not shapes:
        raise SystemExit("no question shape survived slotting; the template pass is broken")

    rng = random.Random(args.seed)
    reports = [r for r in load_reports(args.data_root) if r.identity.report_id not in excluded]
    rng.shuffle(reports)
    if args.reports:
        reports = reports[:args.reports]

    rows = []
    for number, report in enumerate(reports, 1):
        rows.extend(groups_for(report, shapes, companies, args.negatives, rng))
        if number % 100 == 0:
            print(f"{number}/{len(reports)} reports, {len(rows)} rows", flush=True)

    write_jsonl(args.output, rows)
    by_group = defaultdict(list)
    for row in rows:
        by_group[row["id"]].append(row)
    positives = [sum(r["label"] for r in group) for group in by_group.values()]
    print(json.dumps({
        "question_shapes": len(shapes),
        "reports_used": len(reports), "reports_excluded": len(excluded),
        "groups": len(by_group), "rows": len(rows),
        "positives_per_group": round(sum(positives) / max(len(positives), 1), 2),
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
