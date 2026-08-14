#!/usr/bin/env python3
"""Attribute every missed gold table to the stage that lost it.

An aggregate F2 says how much is wrong, never where. Each gold table falls into
exactly one of four states, which point at different work:

* `gate_miss`      - its report never entered the document gate. Fix the question
                     parser or the gate, not the ranker.
* `candidate_miss` - the report was gated but the table never reached the ranked
                     depth. Fix candidate generation: tokenization, matching, recall.
* `rank_miss`      - the table was retrieved but sits below the submitted budget.
                     Fix ranking or the budget; the evidence is already in hand.
* `hit`            - submitted.

Reported per difficulty tier, per label source, and per question size, because a
gain concentrated in easy single-table questions is a different result from the
same gain spread across multi-report ones.
"""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from vifinqa.retrieval import table_budget
from cross_validate_retrieval import build_traces, gold_of
from evaluate_table_retrieval import prefix_score, reciprocal_rank

STATES = ("hit", "rank_miss", "candidate_miss", "gate_miss")


def classify(trace: dict, budget: int, gold: str = "binding") -> dict[str, list[str]]:
    ranked = list(dict.fromkeys(trace["ranked_tables"]))
    submitted, retrieved = set(ranked[:budget]), set(ranked)
    gated = set(trace["selected_docs"])
    states: dict[str, list[str]] = {state: [] for state in STATES}
    for table in gold_of(trace, gold):
        if table in submitted:
            states["hit"].append(table)
        elif table in retrieved:
            states["rank_miss"].append(table)
        elif table.partition("|")[0] in gated:
            states["candidate_miss"].append(table)
        else:
            states["gate_miss"].append(table)
    return states


def share(counter: Counter) -> dict[str, float]:
    total = sum(counter.values())
    return {state: round(counter[state] / total, 4) if total else 0.0 for state in STATES}


def summarize(rows: list[dict]) -> dict:
    counter = Counter()
    for row in rows:
        counter.update({state: len(row["states"][state]) for state in STATES})
    return {
        "questions": len(rows),
        "gold_tables": sum(counter.values()),
        "counts": {state: counter[state] for state in STATES},
        "share": share(counter),
        "mean_f2": round(sum(row["f2"] for row in rows) / len(rows), 4) if rows else 0.0,
        "mean_recall": round(sum(row["recall"] for row in rows) / len(rows), 4) if rows else 0.0,
        "mean_precision": round(sum(row["precision"] for row in rows) / len(rows), 4) if rows else 0.0,
        "mrr": round(sum(row["mrr"] for row in rows) / len(rows), 4) if rows else 0.0,
        "fully_covered": round(sum(1 for row in rows if not any(
            row["states"][state] for state in ("rank_miss", "candidate_miss", "gate_miss")
        )) / len(rows), 4) if rows else 0.0,
    }


def bucket(count: int) -> str:
    return "1 table" if count == 1 else "2 tables" if count == 2 else "3-5 tables" if count <= 5 else "6+ tables"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data" / "raw" / "vifinqa")
    parser.add_argument("--benchmark", type=Path, default=ROOT / "annotations" / "benchmark.jsonl")
    parser.add_argument("--cache", type=Path, default=ROOT / "output" / "diagnostics" / "traces.jsonl")
    parser.add_argument("--table-mode", default="report-coverage")
    parser.add_argument("--depth", type=int, default=50)
    parser.add_argument("--table-top-k", default="auto")
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "diagnostics" / "report.json")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument(
        "--gold", choices=("binding", "full"), default="binding",
        help="'binding' scores the tables named by a row/column binding, which reproduces the live "
             "precision/recall ratio; 'full' scores the restatement-widened set",
    )
    args = parser.parse_args()

    records = {
        json.loads(line)["id"]: json.loads(line)
        for line in args.benchmark.read_text(encoding="utf-8").splitlines() if line.strip()
    }
    if args.cache.exists() and not args.refresh:
        traces = [json.loads(line) for line in args.cache.read_text(encoding="utf-8").splitlines() if line.strip()]
        print(f"reusing {len(traces)} cached traces", file=sys.stderr)
    else:
        traces = build_traces(list(records.values()), args.dataset_root, args.depth, args.table_mode)
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        args.cache.write_text(
            "".join(json.dumps(trace, ensure_ascii=False) + "\n" for trace in traces), encoding="utf-8"
        )

    rows = []
    for trace in traces:
        record = records[trace["id"]]
        budget = table_budget(len(trace["selected_docs"]), args.table_top_k)
        gold_tables = gold_of(trace, args.gold)
        scores = prefix_score(gold_tables, trace["ranked_tables"], budget)
        rows.append({
            "id": trace["id"],
            "tier": record["tier"],
            "source": record["source"],
            "size": bucket(len(gold_tables)),
            "budget": budget,
            "states": classify(trace, budget, args.gold),
            "mrr": reciprocal_rank(gold_tables, trace["ranked_tables"], budget),
            **scores,
        })

    def grouped(key: str) -> dict:
        buckets: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            buckets[row[key]].append(row)
        return {name: summarize(group) for name, group in sorted(buckets.items())}

    report = {
        "table_mode": args.table_mode,
        "table_top_k": args.table_top_k,
        "overall": summarize(rows),
        "by_tier": grouped("tier"),
        "by_source": grouped("source"),
        "by_question_size": grouped("size"),
        "worst_questions": [
            {"id": row["id"], "tier": row["tier"], "f2": round(row["f2"], 4), "budget": row["budget"],
             "missed": {state: row["states"][state] for state in STATES[1:] if row["states"][state]}}
            for row in sorted(rows, key=lambda row: row["f2"])[:15]
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("overall", "by_tier", "by_source", "by_question_size")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
