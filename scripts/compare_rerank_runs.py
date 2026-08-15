#!/usr/bin/env python3
"""Compare two reranker runs, separating what the model did from what it drifted.

`scores_bench_v4.jsonl` and `scores_bench_d100.jsonl` share 11,592
(question, table) pairs whose candidate text is byte-identical, scored by the
same model with the same prompt. 11,454 of them disagree, mean |delta| 0.0098,
max 0.389 — int8 outlier handling and batch packing are not bit-reproducible
across runs. Rebuilding the ranking from the same 50 candidates but the other
run's scores moved benchmark F2 by +0.0076, CI [-0.0063, +0.0235]. That is a
third of the interval this project treats as its resolution, bought by changing
nothing at all, and it is why depth 100 read as +0.0142 when the candidates were
worth +0.0066 of it.

So a run-to-run comparison cannot be read as a method comparison unless the drift
is measured beside it. This script measures both:

* the raw agreement on pairs both runs scored, which is the drift itself;
* an A/A stratum, the questions whose prompt is identical under both settings, so
  their F2 difference is drift and nothing else;
* the treated stratum, the questions the change actually reaches.

For PER_ITEM the split is free: `build_queries` returns a byte-identical prompt
for questions naming zero or one line item, which is 150 of the benchmark's 233,
leaving 83 treated. For a change that touches every question — a different
quantization, say — pass `--aa-items 99` and the whole set becomes the A/A
stratum, which is the right way to measure drift on purpose.

Introduces no new statistics: the fusion is `apply_rerank_scores.fuse` and the
interval is the same paired cluster bootstrap as everywhere else.
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
from apply_rerank_scores import fuse


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def agreement(control: dict, treatment: dict) -> dict:
    """How far the two runs drifted on the pairs they both judged."""
    deltas, identical = [], 0
    for identifier, scores in control.items():
        for table_id, value in scores.items():
            other = treatment.get(identifier, {}).get(table_id)
            if other is None:
                continue
            identical += other == value
            deltas.append(abs(other - value))
    if not deltas:
        return {"shared_pairs": 0}
    return {
        "shared_pairs": len(deltas),
        "identical": identical,
        "identical_share": round(identical / len(deltas), 4),
        "mean_abs_delta": round(sum(deltas) / len(deltas), 6),
        "max_abs_delta": round(max(deltas), 6),
    }


def strata(pairs: dict, ids, aa_items: int) -> tuple[list, list]:
    """Split questions into the ones whose prompt is unchanged and the rest.

    The number of named line items is what decides it, because that is what
    `build_queries` branches on.
    """
    counted = [(i, len(pairs[i].get("line_items") or [])) for i in ids if i in pairs]
    return (
        [i for i, count in counted if count <= aa_items],
        [i for i, count in counted if count > aa_items],
    )


def rank(pairs: dict, scores: dict, mode: str, weight: float, replace_tiers: set, tier_of: dict) -> dict:
    """One ranking per question, under the shipped fusion."""
    ranking = {}
    for identifier, record in pairs.items():
        order = [candidate["table_id"] for candidate in record["candidates"]]
        if identifier in scores:
            question_mode = "replace" if tier_of.get(identifier) in replace_tiers else mode
            order = fuse(record["candidates"], scores[identifier], question_mode, weight)
        ranking[identifier] = order
    return ranking


def score(traces: list[dict], ranking: dict, top_k: str, gold: str) -> dict:
    """Per-question F2 at the shipped budget, over each run's own candidates."""
    rows = {}
    for trace in traces:
        identifier = trace["id"]
        candidates = ranking.get(identifier) or list(dict.fromkeys(trace["ranked_tables"]))
        budget = table_budget(len(trace["selected_docs"]), top_k)
        gold_tables = gold_of(trace, gold)
        rows[identifier] = {
            **prefix_score(gold_tables, candidates, budget),
            "mrr": reciprocal_rank(gold_tables, candidates, budget),
        }
    return rows


def means(rows: dict, ids=None) -> dict:
    selected = [rows[key] for key in (ids if ids is not None else rows)]
    if not selected:
        return {}
    return {
        metric: round(sum(row[metric] for row in selected) / len(selected), 4)
        for metric in ("f2", "recall", "precision", "mrr")
    }


def interval(control: dict, treatment: dict, clusters: dict, ids: list, iterations: int) -> dict:
    if not ids:
        return {}
    return {
        "questions": len(ids),
        "control": means(control, ids),
        "treatment": means(treatment, ids),
        **{
            key: round(value, 4) for key, value in cluster_bootstrap(
                {i: control[i]["f2"] for i in ids},
                {i: treatment[i]["f2"] for i in ids},
                {i: clusters[i] for i in ids}, iterations,
            ).items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True, help="the pairs file both runs scored")
    parser.add_argument("--control", type=Path, required=True, help="baseline scores.jsonl")
    parser.add_argument("--treatment", type=Path, required=True, help="scores.jsonl under test")
    parser.add_argument("--benchmark", type=Path, default=ROOT / "annotations" / "benchmark.jsonl")
    parser.add_argument("--cache", type=Path, default=ROOT / "output" / "diagnostics" / "traces.jsonl")
    parser.add_argument("--table-top-k", default="auto")
    parser.add_argument("--gold", choices=("binding", "full"), default="binding")
    parser.add_argument("--mode", choices=("fuse", "replace"), default="fuse")
    parser.add_argument("--weight", type=float, default=0.5)
    parser.add_argument("--replace-tiers", default="", help="the shipped tier-conditional rule")
    parser.add_argument("--tiers", type=Path, default=ROOT / "data" / "derived" / "question_tiers.jsonl")
    parser.add_argument(
        "--aa-items", type=int, default=1,
        help="questions naming at most this many line items form the A/A stratum, because "
             "PER_ITEM leaves their prompt byte-identical; pass 99 when the change under test "
             "reaches every question and the whole set is an A/A",
    )
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "diagnostics" / "runs.json")
    args = parser.parse_args()

    records = {record["id"]: record for record in load_jsonl(args.benchmark)}
    traces = [trace for trace in load_jsonl(args.cache) if trace["id"] in records]
    if any("gold_tables_binding" not in trace for trace in traces) and args.gold == "binding":
        raise SystemExit(f"{args.cache} predates the binding gold definition; rerun --refresh")
    pairs = {record["id"]: record for record in load_jsonl(args.pairs)}
    scores = {}
    for name, path in (("control", args.control), ("treatment", args.treatment)):
        scores[name] = defaultdict(dict)
        for record in load_jsonl(path):
            scores[name][record["id"]].update(record["scores"])

    replace_tiers = {tier.strip() for tier in args.replace_tiers.split(",") if tier.strip()}
    tier_of = {r["id"]: r["tier"] for r in load_jsonl(args.tiers)} if replace_tiers else {}
    ranked = {
        name: rank(pairs, scores[name], args.mode, args.weight, replace_tiers, tier_of)
        for name in scores
    }
    control = score(traces, ranked["control"], args.table_top_k, args.gold)
    treatment = score(traces, ranked["treatment"], args.table_top_k, args.gold)

    cluster_by_report = {
        report: group[0] for group in connected_report_groups(list(records.values())) for report in group
    }
    clusters = {trace["id"]: cluster_of(trace, cluster_by_report) for trace in traces}

    aa, treated = strata(pairs, list(control), args.aa_items)
    by_tier = defaultdict(list)
    for identifier in control:
        by_tier[records[identifier]["tier"]].append(identifier)

    report = {
        "control": str(args.control),
        "treatment": str(args.treatment),
        "gold": args.gold,
        "fusion": {"mode": args.mode, "weight": args.weight, "replace_tiers": sorted(replace_tiers)},
        "agreement": agreement(scores["control"], scores["treatment"]),
        "overall": interval(control, treatment, clusters, list(control), args.iterations),
        # The A/A stratum is the floor: whatever it reads, the treated stratum has
        # to beat it before any of the difference can be called the method.
        "aa_stratum": {"named_items_at_most": args.aa_items,
                       **interval(control, treatment, clusters, aa, args.iterations)},
        "treated_stratum": {"named_items_above": args.aa_items,
                            **interval(control, treatment, clusters, treated, args.iterations)},
        "by_tier": {
            tier: {
                "questions": len(ids),
                "control_f2": means(control, ids)["f2"],
                "treatment_f2": means(treatment, ids)["f2"],
                "delta": round(means(treatment, ids)["f2"] - means(control, ids)["f2"], 4),
            }
            for tier, ids in sorted(by_tier.items())
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
