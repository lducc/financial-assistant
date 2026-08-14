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

from vifinqa.listwise import agreement, borda, splice


def load_orders(path: Path) -> dict[str, list[str]]:
    return {
        str(record["id"]): record["order"]
        for record in map(json.loads, path.read_text(encoding="utf-8").splitlines())
        if record
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranking", type=Path, required=True, help="the ranking to reorder the head of")
    parser.add_argument(
        "--orders", type=Path, action="append", required=True,
        help="orders.jsonl from rank_listwise.py; repeat once per presentation, and "
             "the passes are merged by Borda count so a candidate has to rank well "
             "under every presentation rather than just the one it was shown first in",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "rerank" / "ranking_listwise.json")
    args = parser.parse_args()

    ranking = json.loads(args.ranking.read_text(encoding="utf-8"))
    passes = [load_orders(path) for path in args.orders]

    spliced, moved, untouched, empty = {}, 0, 0, 0
    correlations = []
    for identifier, order in ranking.items():
        heads = [pass_[identifier] for pass_ in passes if pass_.get(identifier)]
        if len(heads) > 1:
            correlations.append(agreement(heads[0], heads[1]))
        head = borda(heads) if len(heads) > 1 else (heads[0] if heads else [])
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
    summary = {
        "output": str(args.output),
        "passes": len(passes),
        "questions": len(spliced),
        "ordered": len(spliced) - empty,
        "no_order_kept_as_is": empty,
        "top_table_changed": moved,
        "ranking_unchanged": untouched,
    }
    if correlations:
        # Near 1 means the model ranked the same way whichever order it saw, so a
        # flat score means it cannot separate these tables. Near 0 means it
        # followed the presentation, and the run says nothing about the tables.
        summary["mean_pass_agreement"] = round(sum(correlations) / len(correlations), 4)
        summary["questions_compared"] = len(correlations)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
