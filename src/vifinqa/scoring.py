"""Retrieval metrics, gold definitions, and the paired interval.

These were defined in `scripts/evaluate_table_retrieval.py` and
`scripts/cross_validate_retrieval.py` and imported back out of them by seven
other scripts, which is why every one of those scripts had to put `scripts/` on
`sys.path` before it could import anything. They are pure functions over lists
and dicts with no CLI around them, so they belong in the package and the scripts
keep only their command line.

One rule holds all of it together and is the reason to keep these in one place:
a comparison is only readable if both sides use the same gold definition, the
same prefix length, and the same clustering. A second copy of any of the three
is a comparison that can silently stop measuring what it claims to.
"""

from collections import defaultdict
import math
import random


# Fixed so a bootstrap interval is reproducible across runs and machines.
SEED = 20260812


def unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def gold_tables_for(annotation: dict, gold: str = "binding") -> list[str]:
    """The gold set to score against, under one of two definitions.

    `complete_benchmark_labels.py` widened gold with restatements — the same
    figure repeated in a note, the cash-flow, the equity movement table — taking
    it from 2.57 to 4.50 tables per question. Live says that overshot. Submitting
    k tables against G gold gives precision/recall = G/k, and the live ratio is
    0.563; the widened set gives 0.764 at the same budget while the tables named
    by a row/column binding give 0.572. So `binding` is what the organizers
    score, and it leads.

    `full` is kept because the gap between the two is what exposed the problem,
    and because a change that helps one and hurts the other is worth seeing.
    """
    if gold == "full":
        return annotation["gold_tables"]
    bound = unique([
        binding["table"] for binding in annotation.get("row_column_bindings", [])
        if binding.get("table")
    ])
    # A record with no bindings has nothing to narrow to; scoring it against an
    # empty set would silently count it as a total miss.
    return bound or annotation["gold_tables"]


def gold_of(trace: dict, gold: str) -> list[str]:
    """The same choice, for a cached trace rather than an annotation.

    Traces cached before the binding-only definition existed carry only the wide
    set; falling back to it keeps them readable, and the caller warns so the
    stale cache is refreshed rather than quietly scored against the wrong gold.
    """
    if gold == "full":
        return trace["gold_tables"]
    return trace.get("gold_tables_binding") or trace["gold_tables"]


def prefix_score(gold_tables: list[str], ranked_tables: list[str], k: int) -> dict:
    gold = set(gold_tables)
    ranked = unique(ranked_tables)[:k]
    hits = len(gold & set(ranked))
    precision = hits / k if k else 0.0
    recall = hits / len(gold) if gold else 0.0
    f2 = 5 * precision * recall / (4 * precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f2": f2}


def reciprocal_rank(gold_tables: list[str], ranked_tables: list[str], k: int) -> float:
    gold = set(gold_tables)
    for rank, table_id in enumerate(unique(ranked_tables)[:k], 1):
        if table_id in gold:
            return 1 / rank
    return 0.0


def reorder(trace: dict, ranking: dict[str, list[str]]) -> dict:
    """Re-sort a cached trace's candidates into a reranked order.

    A reranker only permutes; it cannot retrieve. Intersecting the ranking with
    the candidates the trace already holds enforces that, so a table the model
    named but that retrieval never surfaced cannot enter, and the four-state
    attribution keeps meaning what it says: `candidate_miss` stays a retrieval
    failure rather than becoming a ranking one. Candidates the ranking omits keep
    their retrieved order behind the ones it named.
    """
    order = ranking.get(str(trace["id"]))
    if not order:
        return trace
    candidates = unique(trace["ranked_tables"])
    named = [table for table in unique(order) if table in set(candidates)]
    rest = [table for table in candidates if table not in set(named)]
    return {**trace, "ranked_tables": named + rest}


def connected_report_groups(records: list[dict]) -> list[tuple[str, ...]]:
    """Reports joined by any question that cites them together.

    Questions sharing evidence are not independent, so the bootstrap resamples
    these groups rather than questions, and cross-validation blocks folds on
    them so no report is in both a fold's training and evaluation view.
    """
    parent: dict[str, str] = {}

    def find(report: str) -> str:
        parent.setdefault(report, report)
        if parent[report] != report:
            parent[report] = find(parent[report])
        return parent[report]

    def join(left: str, right: str) -> None:
        left, right = find(left), find(right)
        if left != right:
            parent[max(left, right)] = min(left, right)

    for record in records:
        reports = sorted(set(record["annotation"]["gold_reports"]))
        for report in reports:
            find(report)
        for report in reports[1:]:
            join(reports[0], report)
    groups: dict[str, list[str]] = defaultdict(list)
    for report in parent:
        groups[find(report)].append(report)
    return sorted(tuple(sorted(group)) for group in groups.values())


def cluster_of(trace: dict, cluster_by_report: dict[str, str]) -> str:
    return cluster_by_report.get(trace["gold_reports"][0], trace["gold_reports"][0])


def clusters_for(traces: list[dict], records: list[dict]) -> dict[int, str]:
    """Question ID to report cluster, which is what the bootstrap resamples.

    Every caller built this in the same three lines; the two halves have to agree
    or the pairing is wrong, so they are one call.
    """
    cluster_by_report = {
        report: group[0] for group in connected_report_groups(records) for report in group
    }
    return {trace["id"]: cluster_of(trace, cluster_by_report) for trace in traces}


def mean(values: list[float], weights: list[float] | None = None) -> float:
    """Weighted mean, so a sample that over-represents hard questions still
    estimates corpus performance. Per-tier figures pass no weights."""
    if not values:
        return 0.0
    if weights is None:
        return sum(values) / len(values)
    total = sum(weights)
    return sum(v * w for v, w in zip(values, weights)) / total if total else 0.0


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


def metric_means(rows: dict[int, dict], ids: list[int] | None = None) -> dict[str, float]:
    """Mean of each metric over the chosen questions, or all of them."""
    selected = [rows[key] for key in (ids if ids is not None else rows)]
    if not selected:
        return {}
    return {
        metric: round(sum(row[metric] for row in selected) / len(selected), 4)
        for metric in ("f2", "recall", "precision", "mrr")
    }
