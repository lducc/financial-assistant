#!/usr/bin/env python3
"""Promote the candidates whose account code is the one the question asks for.

The pool holds 87.4% of the gold and the shipped order surfaces 61.5% of it, so
ordering is worth 0.22 F2 and retrieval is worth nothing. This is the first
attempt at the ordering: Circular 200 files every line item under a fixed `Mã
số`, the lexicon says which, and a primary statement either carries that code or
does not. A promoted candidate keeps its place relative to the other promoted
ones, so this only lifts matches over non-matches and never reshuffles either
group.

Only statements are promoted. Notes carry no code column, and the note hop was
measured at 0.05 precision, so there is nothing to promote them on.
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_item_expansion import question_items
from vifinqa.jsonl import load_jsonl
from vifinqa.lexicon import load_lexicon, resolve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranking", type=Path, required=True)
    parser.add_argument("--questions", type=Path, default=ROOT / "data/raw/vifinqa/questions/questions.jsonl")
    parser.add_argument("--structure", type=Path, default=ROOT / "data/derived/table_structure.jsonl")
    parser.add_argument("--lexicon", type=Path, default=ROOT / "data/derived/account_lexicon.json")
    parser.add_argument("--limit", type=int, default=1, help="codes kept per line item")
    parser.add_argument("--top", type=int, default=2, help="most candidates to promote")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    labels = load_lexicon(args.lexicon)
    codes_of = {row["id"]: set(row["codes"]) for row in load_jsonl(args.structure) if row["kind"] == "statement"}
    ranking = json.loads(args.ranking.read_text(encoding="utf-8"))

    boosted, moved, promotions = {}, 0, 0
    for question in load_jsonl(args.questions):
        order = ranking.get(str(question["id"]))
        if not order:
            continue
        wanted = {code for item in question_items(question["question"])
                  for code in resolve(item, labels, args.limit)}
        matches = [table for table in order if wanted & codes_of.get(table, set())][:args.top]
        new = matches + [table for table in order if table not in matches]
        moved += new[:3] != order[:3]
        promotions += len(matches)
        boosted[str(question["id"])] = new

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(boosted, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "questions": len(boosted), "head_changed": moved,
        "promotions_per_question": round(promotions / max(len(boosted), 1), 2),
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
