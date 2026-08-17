#!/usr/bin/env python3
"""Reorder candidates by how many of their neighbours report the same figures.

A filing is internally redundant by regulation: a figure printed in a primary
statement is printed again in the note that details it and in the following
year's comparative column. So a number carried by several of a question's
candidates is a reported figure, and the tables carrying it are its evidence set,
while a table agreeing with nothing is usually a listing or a schedule that
merely mentions the wording.

The signal needs no labels and no model — it is arithmetic over the candidate
text — and it holds for any corpus where the same figures are restated across
statements, which is what a filing standard requires.

Agreement is fused with the existing order by reciprocal rank, so a table has to
be both plausible to the ranker and corroborated by its neighbours to move up.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from vifinqa.fusion import RRF_OFFSET
from vifinqa.jsonl import load_jsonl
from vifinqa.statements import rows as parse_rows

DIGITS = re.compile(r"\d")


def figures(text: str) -> set[str]:
    """Amounts a table prints, as digit strings, ignoring anything too short."""
    found = set()
    for row in parse_rows(text) or [[line] for line in text.split("\n")]:
        for cell in row:
            digits = "".join(DIGITS.findall(cell)).lstrip("0")
            if len(digits) >= 7:
                found.add(digits)
    return found


def agreement(candidates: list[dict], depth: int) -> Counter:
    """How many other candidates print at least one of this table's figures."""
    seen = {c["table_id"]: figures(c["text"]) for c in candidates[:depth]}
    return Counter({
        table: sum(1 for other, values in seen.items() if other != table and values & mine)
        for table, mine in seen.items()
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--ranking", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--depth", type=int, default=20, help="candidates that can corroborate")
    parser.add_argument("--weight", type=float, default=0.3, help="share of the fused score agreement carries")
    args = parser.parse_args()

    ranking = json.loads(args.ranking.read_text(encoding="utf-8"))
    out, moved = {}, 0
    for record in load_jsonl(args.pairs):
        order = ranking.get(str(record["id"]))
        if not order:
            continue
        peers = agreement(record["candidates"], args.depth)
        by_peers = sorted(peers, key=lambda t: -peers[t])
        place = {table: index for index, table in enumerate(by_peers)}
        score = {
            table: (1 - args.weight) / (RRF_OFFSET + index + 1)
                   + args.weight / (RRF_OFFSET + place.get(table, len(order)) + 1)
            for index, table in enumerate(order)
        }
        new = sorted(order, key=lambda t: -score[t])
        moved += new[:3] != order[:3]
        out[str(record["id"])] = new

    args.output.write_text(json.dumps(out, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"questions": len(out), "head_changed": moved, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
