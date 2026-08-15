#!/usr/bin/env python3
"""Build cross-encoder training pairs from labelled questions and the candidate pool.

An oracle reordering of the candidates we already retrieve scores 0.8020 on the
benchmark against 0.6549 shipped, so 0.147 F2 sits inside the ranking stage and
nowhere else. Zero-shot Qwen3-Reranker captures part of it; the literature's one
reported edge on financial tables (HiREC) came from fine-tuning the cross-encoder,
and the organizers permit it.

Positives are the labelled gold tables that actually appear in the candidate pool
— a gold table retrieval never surfaced cannot teach a ranker anything, and
counting it would silently inflate the positive rate. Negatives are the
highest-ranked non-gold candidates, because those are the ones the model has to
push down; sampling the tail instead would train on comparisons the ranking never
has to make.

The emitted query and document strings are exactly what `kaggle/rerank_qwen_8b.py`
builds at inference, so training and scoring see the same text. Keep them in sync:
a prompt that differs between the two is a silent domain shift.
"""

import argparse
from collections import Counter
import json
from pathlib import Path

from vifinqa.jsonl import load_jsonl
from vifinqa.scoring import gold_tables_for

ROOT = Path(__file__).resolve().parents[1]


def build_queries(record: dict, per_item: bool) -> list[str]:
    """The query strings `rerank_qwen_8b.build_queries` would produce."""
    items = record.get("line_items") or []
    if not items:
        return [record["question"]]
    if per_item and len(items) > 1:
        return [f"{record['question']}\nChỉ tiêu cần tìm: {item}" for item in items]
    return [f"{record['question']}\nChỉ tiêu cần tìm: {'; '.join(items)}"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labels", type=Path, default=ROOT / "annotations" / "train" / "accepted.jsonl",
        help="labelled questions to train on; must be disjoint from whatever is used to evaluate",
    )
    parser.add_argument("--pairs", type=Path, default=ROOT / "output" / "rerank" / "pairs_v4.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "rerank" / "training.jsonl")
    parser.add_argument(
        "--negatives", type=int, default=8,
        help="hard negatives per question, taken from the best-ranked non-gold candidates",
    )
    parser.add_argument("--per-item", action="store_true", help="one query per named line item, as PER_ITEM=1 does")
    parser.add_argument(
        "--gold", choices=("binding", "full"), default="binding",
        help="'binding' trains on the tables named by a row/column binding, which is the "
             "definition whose cardinality matches the organizers' (3.24 against 3.29)",
    )
    parser.add_argument(
        "--exclude", type=Path, action="append", default=[],
        help="label file whose question IDs must not appear in the training set; repeat per file",
    )
    args = parser.parse_args()

    def load(path: Path) -> list[dict]:
        return load_jsonl(path)

    excluded = {record["id"] for path in args.exclude for record in load(path)}
    labels = {record["id"]: record for record in load(args.labels) if record["id"] not in excluded}
    pairs = {record["id"]: record for record in load(args.pairs)}

    rows, stats = [], Counter()
    unreachable = []
    for identifier, record in sorted(labels.items()):
        pool = pairs.get(identifier)
        if pool is None:
            stats["no_candidate_pool"] += 1
            continue
        gold = set(gold_tables_for(record["annotation"], args.gold))
        candidates = pool["candidates"]
        positives = [c for c in candidates if c["table_id"] in gold]
        # Gold that retrieval never surfaced is a candidate-generation failure and
        # has no document text to train on; count it so the gap stays visible.
        stats["gold_out_of_pool"] += len(gold) - len(positives)
        if not positives:
            unreachable.append(identifier)
            stats["questions_without_a_positive"] += 1
            continue
        negatives = [c for c in candidates if c["table_id"] not in gold][:args.negatives]
        for query in build_queries(pool, args.per_item):
            for candidate, label in [(c, 1) for c in positives] + [(c, 0) for c in negatives]:
                rows.append({
                    "id": identifier,
                    "query": query,
                    "document": candidate["text"],
                    "label": label,
                    "table_id": candidate["table_id"],
                })
        stats["questions"] += 1
        stats["positives"] += len(positives)
        stats["negatives"] += len(negatives)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output),
        "labels": str(args.labels),
        "excluded_ids": len(excluded),
        "questions_trained": stats["questions"],
        "questions_without_a_positive": stats["questions_without_a_positive"],
        "gold_tables_out_of_pool": stats["gold_out_of_pool"],
        "positives": stats["positives"],
        "negatives": stats["negatives"],
        "rows": len(rows),
        "positive_rate": round(stats["positives"] / max(1, stats["positives"] + stats["negatives"]), 4),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
