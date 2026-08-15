#!/usr/bin/env python3
"""Compare two rankings over one cached retrieval, with a paired interval.

A ranking is a permutation of candidates retrieval already produced, so two
rankings can be scored against the same traces and the difference read per
question. That pairing is what makes the interval tight enough to decide
anything: the benchmark's 233 records sit in 184 report clusters, and an
unpaired comparison spends most of its resolution on between-question variance
that both rankings share.

The decision rule is `docs/benchmark.md`'s and is not restated here to be
softened: a change is accepted only when the mean improves, the bootstrap CI on
the delta excludes zero, and no difficulty tier regresses. This script reports
all three; it does not decide.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from vifinqa.retrieval import table_budget
from cross_validate_retrieval import cluster_bootstrap, cluster_of, gold_of
from evaluate_table_retrieval import connected_report_groups, prefix_score, reciprocal_rank
from diagnose_retrieval import reorder


def scored(traces: list[dict], ranking: dict[str, list[str]] | None, top_k: str, gold: str) -> dict[int, dict]:
    """Per-question scores for one ranking, at the shipped budget."""
    rows = {}
    for trace in traces:
        ordered = reorder(trace, ranking) if ranking else trace
        budget = table_budget(len(trace["selected_docs"]), top_k)
        gold_tables = gold_of(trace, gold)
        rows[trace["id"]] = {
            **prefix_score(gold_tables, ordered["ranked_tables"], budget),
            "mrr": reciprocal_rank(gold_tables, ordered["ranked_tables"], budget),
        }
    return rows


def means(rows: dict[int, dict], keys: list[int] | None = None) -> dict[str, float]:
    selected = [rows[key] for key in (keys if keys is not None else rows)]
    if not selected:
        return {}
    return {
        metric: round(sum(row[metric] for row in selected) / len(selected), 4)
        for metric in ("f2", "recall", "precision", "mrr")
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, default=ROOT / "annotations" / "benchmark.jsonl")
    parser.add_argument("--cache", type=Path, default=ROOT / "output" / "diagnostics" / "traces.jsonl")
    parser.add_argument(
        "--baseline", type=Path,
        help="ranking.json to compare against; the retrieved sparse order if omitted",
    )
    parser.add_argument("--candidate", type=Path, required=True, help="ranking.json under test")
    parser.add_argument("--table-top-k", default="auto")
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument(
        "--gold", choices=("binding", "full"), default="binding",
        help="'binding' scores the tables named by a row/column binding, which reproduces the live "
             "precision/recall ratio; 'full' scores the restatement-widened set",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "diagnostics" / "comparison.json")
    args = parser.parse_args()

    records = {
        json.loads(line)["id"]: json.loads(line)
        for line in args.benchmark.read_text(encoding="utf-8").splitlines() if line.strip()
    }
    traces = [
        json.loads(line) for line in args.cache.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if any("gold_tables_binding" not in trace for trace in traces) and args.gold == "binding":
        raise SystemExit(
            f"{args.cache} predates the binding gold definition; rerun "
            "scripts/diagnose_retrieval.py --refresh before comparing"
        )
    load = lambda path: json.loads(path.read_text(encoding="utf-8")) if path else None
    baseline = scored(traces, load(args.baseline), args.table_top_k, args.gold)
    candidate = scored(traces, load(args.candidate), args.table_top_k, args.gold)

    cluster_by_report = {
        report: group[0] for group in connected_report_groups(list(records.values())) for report in group
    }
    clusters = {trace["id"]: cluster_of(trace, cluster_by_report) for trace in traces}

    by_tier = defaultdict(list)
    for trace in traces:
        by_tier[records[trace["id"]]["tier"]].append(trace["id"])
    by_source = defaultdict(list)
    for trace in traces:
        by_source[records[trace["id"]].get("source", "unsplit")].append(trace["id"])

    report = {
        "baseline": str(args.baseline) if args.baseline else "sparse (retrieved order)",
        "candidate": str(args.candidate),
        "gold": args.gold,
        "questions": len(traces),
        "clusters": len(set(clusters.values())),
        "overall": {
            "baseline": means(baseline),
            "candidate": means(candidate),
            **{
                key: round(value, 4) for key, value in cluster_bootstrap(
                    {i: row["f2"] for i, row in baseline.items()},
                    {i: row["f2"] for i, row in candidate.items()},
                    clusters, args.iterations,
                ).items()
            },
        },
        # Per tier and per label source, because the rule forbids accepting a
        # gain that hides a regression, and because v3 records were found without
        # a retriever in the loop and so read as the less flattering half.
        "by_tier": {
            tier: {
                "questions": len(ids),
                "baseline_f2": means(baseline, ids)["f2"],
                "candidate_f2": means(candidate, ids)["f2"],
                "delta": round(means(candidate, ids)["f2"] - means(baseline, ids)["f2"], 4),
            }
            for tier, ids in sorted(by_tier.items())
        },
        "by_source": {
            source: {
                "questions": len(ids),
                "baseline_f2": means(baseline, ids)["f2"],
                "candidate_f2": means(candidate, ids)["f2"],
                "delta": round(means(candidate, ids)["f2"] - means(baseline, ids)["f2"], 4),
            }
            for source, ids in sorted(by_source.items())
        },
        "questions_improved": sum(1 for i in baseline if candidate[i]["f2"] > baseline[i]["f2"]),
        "questions_regressed": sum(1 for i in baseline if candidate[i]["f2"] < baseline[i]["f2"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
