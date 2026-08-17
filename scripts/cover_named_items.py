#!/usr/bin/env python3
"""Reorder a ranking so the first tables cover every line item the question names.

Relevance ranking answers "which table is most about this question", and a
question naming two items gets four tables about the first and one about the
second. Cutting to a budget then drops the second item entirely, which is why our
recall needs eight tables to reach what synera reaches in three and a half.

Coverage is the standard fix — MMR (Carbonell and Goldstein 1998), submodular
selection (Lin and Bilmes 2011) — and here it needs no scores at all: a table
either carries the item in its own first column or it does not. The greedy pass
promotes the best-ranked carrier of each named item in turn, then leaves the rest
of the order alone, so it can only move tables the ranker already proposed.
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_item_expansion import index_corpus, question_items
from vifinqa.jsonl import load_jsonl


def cover(order: list[str], items: list[str], carriers: dict[str, set[str]]) -> list[str]:
    """Promote one carrier per named item, keeping the ranker's order among them."""
    first = []
    for item in items:
        promoted = next((table for table in order
                         if table in carriers.get(item, ()) and table not in first), None)
        if promoted:
            first.append(promoted)
    first.sort(key=order.index)
    return first + [table for table in order if table not in first]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranking", type=Path, required=True)
    parser.add_argument("--questions", type=Path, default=ROOT / "data/raw/vifinqa/questions/questions.jsonl")
    parser.add_argument("--corpus", type=Path, default=ROOT / "data/raw/vifinqa")
    parser.add_argument("--catalog", type=Path, default=ROOT / "data/derived/table_catalog/tables.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    questions = list(load_jsonl(args.questions))
    items = {question["id"]: sorted(question_items(question["question"])) for question in questions}
    carriers = index_corpus(args.catalog, args.corpus, set().union(*items.values()))

    ranking = json.loads(args.ranking.read_text(encoding="utf-8"))
    covered = {}
    moved = 0
    for question in questions:
        order = ranking.get(str(question["id"]))
        if not order:
            continue
        new = cover(order, items[question["id"]], carriers)
        moved += new[:3] != order[:3]
        covered[str(question["id"])] = new

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(covered, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "questions": len(covered),
        "head_changed": moved,
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
