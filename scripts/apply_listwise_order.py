#!/usr/bin/env python3
"""Splice listwise orders into the head of an existing ranking.

The listwise pass only sees the top 20, so the tail keeps whatever the pointwise
stage decided. Nothing is dropped and nothing new is admitted: a table the model
named but that was never retrieved cannot enter the submission, and a candidate
the model omitted keeps its incoming position behind the ones it named.

Unlike the pointwise scores there is nothing to fuse here — the model's output is
an order, not a signal to combine — so this is a splice rather than a rank
fusion.
"""

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifinqa.listwise import splice


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranking", type=Path, required=True, help="the ranking to reorder the head of")
    parser.add_argument("--orders", type=Path, required=True, help="orders.jsonl from rank_listwise.py")
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "rerank" / "ranking_listwise.json")
    args = parser.parse_args()

    ranking = json.loads(args.ranking.read_text(encoding="utf-8"))
    orders = {
        str(record["id"]): record["order"]
        for record in map(json.loads, args.orders.read_text(encoding="utf-8").splitlines())
        if record
    }

    spliced, moved, untouched, empty = {}, 0, 0, 0
    for identifier, order in ranking.items():
        head = orders.get(identifier) or []
        if not head:
            empty += 1
            spliced[identifier] = order
            continue
        result = splice(order, head)
        spliced[identifier] = result
        moved += result[0] != order[0]
        untouched += result == order
        assert sorted(result) == sorted(order), f"id={identifier} lost or gained a table"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(spliced, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "questions": len(spliced),
        "ordered": len(spliced) - empty,
        "no_order_kept_as_is": empty,
        "top_table_changed": moved,
        "ranking_unchanged": untouched,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
