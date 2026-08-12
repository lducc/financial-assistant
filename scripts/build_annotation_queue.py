#!/usr/bin/env python3
"""Draw a blind, stratified annotation queue for extending the retrieval benchmark.

Two constraints shape the sample. First, difficulty: the existing 150 records
give a cluster-bootstrap CI half-width of 0.113 F2 on the hard tier, so nothing
about hard questions is currently measurable; the queue is weighted toward hard
and intermediate. Second, independence: a new question whose gated reports
already appear in gold-150 shares a bootstrap cluster with it, which adds records
without adding information, so those are excluded.

The queue carries the question and nothing else. Reviewers must not see retrieval
output, candidate tables, or existing labels, per docs/v2_annotation_protocol.md.
"""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SEED = 20260812
# Hard and intermediate carry the widest intervals and the worst retrieval, so
# they get the sampling weight.
TIER_WEIGHTS = {"hard": 0.40, "intermediate": 0.30, "medium": 0.20, "easy": 0.10}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiers", type=Path, default=ROOT / "data" / "derived" / "question_tiers.jsonl")
    parser.add_argument("--existing", type=Path, default=ROOT / "annotations" / "gold_150.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "annotations" / "v3" / "queue.jsonl")
    parser.add_argument("--count", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument(
        "--labelled", type=Path, action="append", default=[],
        help="already-annotated label files to keep out of new batches; repeatable",
    )
    args = parser.parse_args()

    tiers = load_jsonl(args.tiers)
    existing = load_jsonl(args.existing)
    taken = {record["id"] for record in existing}
    for path in args.labelled:
        taken.update(record["id"] for record in load_jsonl(path))
    used_reports = {report for record in existing for report in record["annotation"]["gold_reports"]}

    # Clusters are built from reports, not tickers, so the same company in a
    # different year is already an independent cluster. Exclude only questions
    # whose own gated reports overlap gold-150 evidence.
    eligible = [
        record for record in tiers
        if record["id"] not in taken
        and record["gated_report_ids"]
        and not (set(record["gated_report_ids"]) & used_reports)
    ]
    by_tier: dict[str, list[dict]] = defaultdict(list)
    for record in eligible:
        by_tier[record["tier"]].append(record)

    rng = random.Random(SEED)
    selected: list[dict] = []
    shortfalls = {}
    for tier, weight in TIER_WEIGHTS.items():
        wanted = round(args.count * weight)
        pool = sorted(by_tier.get(tier, []), key=lambda record: record["id"])
        rng.shuffle(pool)
        if len(pool) < wanted:
            shortfalls[tier] = {"wanted": wanted, "available": len(pool)}
        selected.extend(pool[:wanted])

    # Round-robin the tiers into batches so each batch mirrors the stratification.
    # Contiguous slices of an ID-sorted list do not: the first batch drawn that way
    # was almost entirely single-report questions, which flatters any budget policy.
    queues = {tier: [record for record in selected if record["tier"] == tier] for tier in TIER_WEIGHTS}
    for pool in queues.values():
        rng.shuffle(pool)
    ordered: list[dict] = []
    while any(queues.values()):
        for tier in sorted(TIER_WEIGHTS, key=lambda name: -TIER_WEIGHTS[name]):
            if queues[tier]:
                ordered.append(queues[tier].pop())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(
        json.dumps({
            "id": record["id"],
            "question": record["question"],
            "tier": record["tier"],
            "batch": index // args.batch_size,
        }, ensure_ascii=False) + "\n"
        for index, record in enumerate(ordered)
    ), encoding="utf-8")

    print(json.dumps({
        "output": str(args.output),
        "eligible": len(eligible),
        "selected": len(selected),
        "by_tier": dict(Counter(record["tier"] for record in selected)),
        "batches": (len(selected) + args.batch_size - 1) // args.batch_size,
        "excluded_reports": len(used_reports),
        "shortfalls": shortfalls,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
