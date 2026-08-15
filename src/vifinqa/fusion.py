"""Turn cross-encoder scores into the ordering that ships.

Every replacement tried on this task has lost and every fusion that earned its
place has won, so the reranker enters as one more reciprocal rank rather than as
the new order. Lives in the package rather than beside the CLI because
`compare_rerank_runs.py` has to score exactly what ships; two copies of the rule
means a comparison can silently stop measuring the shipped system.

The weight sweep is in docs/research-history.md. Every value in 0.0-0.5 sits
inside the benchmark's resolution, so treat a change as free rather than a gain.
"""

from collections import defaultdict
from pathlib import Path

from vifinqa.jsonl import load_jsonl


RRF_OFFSET = 60


def fuse(
    candidates: list[dict], scores: dict[str, float], mode: str, weight: float = 0.5,
) -> list[str]:
    """Order one question's candidates by weighted reciprocal rank.

    `weight` is the share given to the sparse rank: 1.0 sparse alone, 0.0 the
    model alone, 0.5 the unweighted sum every submission has used.
    """
    sparse = {candidate["table_id"]: candidate["sparse_rank"] for candidate in candidates}
    ranked_by_model = sorted(
        (table_id for table_id in sparse if table_id in scores),
        key=lambda table_id: (-scores[table_id], sparse[table_id]),
    )
    dense_rank = {table_id: rank for rank, table_id in enumerate(ranked_by_model, 1)}
    if mode == "replace":
        return ranked_by_model + [t for t in sorted(sparse, key=sparse.get) if t not in dense_rank]
    return sorted(
        sparse,
        key=lambda table_id: (
            -(
                weight / (RRF_OFFSET + sparse[table_id])
                + ((1 - weight) / (RRF_OFFSET + dense_rank[table_id]) if table_id in dense_rank else 0.0)
            ),
            sparse[table_id],
        ),
    )


def load_scores(paths: list[Path]) -> dict[int, dict[str, float]]:
    """Merge score files, inside the per-question dict as well as across it.

    Sharded runs never share a question, but a depth extension does and differs
    only in which candidates it judged, so the union has to reach inside.
    """
    scored: dict[int, dict[str, float]] = defaultdict(dict)
    for path in paths:
        for record in load_jsonl(path):
            scored[record["id"]].update(record["scores"])
    return scored


def rankings(
    pairs: dict, scored: dict, mode: str, weight: float, replace_tiers: set, tier_of: dict,
) -> dict[str, list[str]]:
    """The shipped ordering rule, keyed by question ID as ranking.json is."""
    ranking = {}
    for identifier, record in pairs.items():
        order = [candidate["table_id"] for candidate in record["candidates"]]
        if identifier in scored:
            question_mode = "replace" if tier_of.get(identifier) in replace_tiers else mode
            order = fuse(record["candidates"], scored[identifier], question_mode, weight)
        ranking[str(identifier)] = order
    return ranking
