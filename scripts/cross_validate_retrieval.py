#!/usr/bin/env python3
"""Cross-validate table-selection policies against verified retrieval labels.

Retrieval to a fixed depth is policy-independent: the budget, the selection rule,
and any rescoring applied afterwards all operate on the same ranked list. So the
expensive part runs once and is cached, and every policy is then scored offline
over group-blocked folds. Folds are blocked by connected report groups, so no
report contributes to both a fold's training and evaluation view, and questions
sharing evidence cannot leak across folds.

Point estimates on 150 records cannot separate a real gain from noise, so every
comparison reports a paired cluster bootstrap interval against the baseline
policy alongside the per-fold spread.
"""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from docs import load_companies, load_reports as load_doc_reports, parse_question, required_report_years, retrieve_docs
from vifinqa.retrieval import load_reports, retrieve_rows, table_budget
from evaluate_table_retrieval import connected_report_groups, prefix_score, reciprocal_rank, split_records


SEED = 20260812
TIER_MULTIPLIER = {"easy": 1, "medium": 2, "intermediate": 3, "hard": 3}
# Multipliers apply to the gated report count directly, never to table_budget():
# that helper carries whichever multiplier currently ships, so routing policies
# through it would silently rename every row whenever the default changes.
POLICIES = {
    "fixed-5": lambda reports, tier: 5,
    "fixed-10": lambda reports, tier: 10,
    "reports x1": lambda reports, tier: min(30, max(1, reports)),
    "reports x2": lambda reports, tier: min(30, max(1, 2 * reports)),
    "reports x3": lambda reports, tier: min(30, max(1, 3 * reports)),
    "reports x4": lambda reports, tier: min(30, max(1, 4 * reports)),
    "shipped default": lambda reports, tier: table_budget(reports),
    # Easy questions have exactly one gold table in one report, so spending extra
    # slots on them only costs precision. Difficulty comes from the tier
    # classifier, which is validated against the organizer's published counts.
    "tier-aware": lambda reports, tier: min(30, max(1, TIER_MULTIPLIER.get(tier, 3) * reports)),
}


def build_traces(labels: list[dict], dataset_root: Path, depth: int, mode: str, reranker: str | None = None) -> list[dict]:
    companies = load_companies(dataset_root / "code_stock.csv")
    doc_reports = load_doc_reports(dataset_root / "financial_statements")
    table_reports = load_reports(dataset_root)
    traces = []
    for number, record in enumerate(labels, 1):
        parsed = parse_question(record["question"], companies)
        if not parsed.tickers and parsed.candidate_tickers:
            parsed.tickers = parsed.candidate_tickers[:1]
        docs, _ = retrieve_docs(parsed, doc_reports)
        retrieval = retrieve_rows(
            record["question"],
            {
                "tickers": parsed.tickers,
                "years": parsed.years,
                "slot_years": required_report_years(parsed),
                "scope": parsed.scope,
            },
            table_reports,
            top_k=depth,
            report_ids=docs,
            mode=mode,
            reranker=reranker,
            reranker_batch_size=32,
        )
        traces.append({
            "id": record["id"],
            "gold_tables": record["annotation"]["gold_tables"],
            "gold_reports": sorted(set(record["annotation"]["gold_reports"])),
            "selected_docs": docs,
            "ranked_tables": [table["table_id"] for table in retrieval["tables"]],
            # Scores let cutoff policies be swept offline, like budgets already are.
            "ranked_scores": [table["score"] for table in retrieval["tables"]],
            "latency_ms": retrieval["latency_ms"],
        })
        if number % 25 == 0:
            print(f"retrieved {number}/{len(labels)}", flush=True, file=sys.stderr)
    return traces


def cluster_of(trace: dict, cluster_by_report: dict[str, str]) -> str:
    return cluster_by_report.get(trace["gold_reports"][0], trace["gold_reports"][0])


def score(trace: dict, budget: int) -> dict:
    scores = prefix_score(trace["gold_tables"], trace["ranked_tables"], budget)
    scores["mrr"] = reciprocal_rank(trace["gold_tables"], trace["ranked_tables"], budget)
    scores["k"] = budget
    return scores


def mean(values: list[float], weights: list[float] | None = None) -> float:
    """Weighted mean, so a sample that over-represents hard questions still
    estimates corpus performance. Per-tier figures pass no weights."""
    if not values:
        return 0.0
    if weights is None:
        return sum(values) / len(values)
    total = sum(weights)
    return sum(v * w for v, w in zip(values, weights)) / total if total else 0.0


def fold_of(cluster: str, folds: int) -> int:
    return int.from_bytes(cluster.encode("utf-8"), "big") % folds


def cluster_bootstrap(
    baseline: dict[int, float], candidate: dict[int, float], clusters: dict[int, str], iterations: int,
) -> dict[str, float]:
    """Paired bootstrap over report clusters, matching the v2 evaluation protocol."""
    grouped: dict[str, list[int]] = defaultdict(list)
    for identifier, cluster in clusters.items():
        grouped[cluster].append(identifier)
    groups = list(grouped.values())
    rng = random.Random(SEED)
    deltas = []
    for _ in range(iterations):
        sampled = [identifier for _ in groups for identifier in rng.choice(groups)]
        deltas.append(mean([candidate[identifier] - baseline[identifier] for identifier in sampled]))
    deltas.sort()
    return {
        "delta": mean([candidate[identifier] - baseline[identifier] for identifier in baseline]),
        "ci95_low": deltas[int(0.025 * iterations)],
        "ci95_high": deltas[min(iterations - 1, math.ceil(0.975 * iterations) - 1)],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data" / "raw" / "vifinqa")
    parser.add_argument("--labels", type=Path, default=ROOT / "annotations" / "gold_150.jsonl")
    parser.add_argument("--tiers", type=Path, default=ROOT / "data" / "derived" / "question_tiers.jsonl")
    parser.add_argument("--cache", type=Path, default=ROOT / "output" / "cv" / "traces.jsonl")
    parser.add_argument("--table-mode", default="report-coverage")
    parser.add_argument("--reranker", choices=("mmarco",), help="cross-encoder rerank of the ranking head")
    parser.add_argument("--depth", type=int, default=50)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--baseline-policy", default="fixed-5")
    parser.add_argument("--refresh", action="store_true", help="ignore cached traces and retrieve again")
    parser.add_argument(
        "--split", choices=("all", "dev", "test"), default="all",
        help="'test' is the frozen holdout; use it to confirm a decision, never to make one",
    )
    args = parser.parse_args()

    labels = [json.loads(line) for line in args.labels.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.cache.exists() and not args.refresh:
        traces = [json.loads(line) for line in args.cache.read_text(encoding="utf-8").splitlines() if line.strip()]
        print(f"reusing {len(traces)} cached traces from {args.cache}", file=sys.stderr)
    else:
        traces = build_traces(labels, args.dataset_root, args.depth, args.table_mode, args.reranker)
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        args.cache.write_text(
            "".join(json.dumps(trace, ensure_ascii=False) + "\n" for trace in traces), encoding="utf-8"
        )

    if args.split != "all":
        keep = {record["id"] for record in split_records(labels, args.split)}
        labels = [record for record in labels if record["id"] in keep]
        traces = [trace for trace in traces if trace["id"] in keep]

    tiers = {}
    if args.tiers.exists():
        tiers = {
            json.loads(line)["id"]: json.loads(line)["tier"]
            for line in args.tiers.read_text(encoding="utf-8").splitlines() if line.strip()
        }
    # Weights come from the benchmark records themselves when present.
    weight_of = {record["id"]: record.get("weight", 1.0) for record in labels}

    cluster_by_report = {
        report: group[0] for group in connected_report_groups(labels) for report in group
    }
    clusters = {trace["id"]: cluster_of(trace, cluster_by_report) for trace in traces}
    folds = {trace["id"]: fold_of(clusters[trace["id"]], args.folds) for trace in traces}

    results = {}
    per_question: dict[str, dict[int, float]] = {}
    for name, rule in POLICIES.items():
        scored = {
            trace["id"]: score(trace, rule(len(trace["selected_docs"]), tiers.get(trace["id"], "intermediate")))
            for trace in traces
        }
        per_question[name] = {identifier: values["f2"] for identifier, values in scored.items()}
        fold_means = [
            mean([scored[trace["id"]]["f2"] for trace in traces if folds[trace["id"]] == fold])
            for fold in range(args.folds)
        ]
        results[name] = {
            "pooled": {
                metric: round(mean(
                    [scored[trace["id"]][metric] for trace in traces],
                    [weight_of.get(trace["id"], 1.0) for trace in traces],
                ), 4)
                for metric in ("f2", "precision", "recall", "mrr", "k")
            },
            "fold_f2": [round(value, 4) for value in fold_means],
            "fold_mean": round(mean(fold_means), 4),
            "fold_spread": round(max(fold_means) - min(fold_means), 4),
            "by_tier": {
                tier: round(mean([
                    scored[trace["id"]]["f2"] for trace in traces if tiers.get(trace["id"]) == tier
                ]), 4)
                for tier in ("easy", "medium", "intermediate", "hard")
                if any(tiers.get(trace["id"]) == tier for trace in traces)
            },
        }

    baseline = per_question[args.baseline_policy]
    for name in POLICIES:
        if name != args.baseline_policy:
            results[name]["bootstrap_vs_baseline"] = {
                key: round(value, 4)
                for key, value in cluster_bootstrap(baseline, per_question[name], clusters, args.iterations).items()
            }

    print(json.dumps({
        "records": len(traces),
        "clusters": len(set(clusters.values())),
        "folds": args.folds,
        "table_mode": args.table_mode,
        "baseline_policy": args.baseline_policy,
        "policies": results,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
