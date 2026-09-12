from collections import defaultdict
from pathlib import Path

from vifinqa.jsonl import load_jsonl


RRF_OFFSET = 60


def reciprocal(rank: int | None) -> float:
    return 1.0 / (RRF_OFFSET + rank) if rank else 0.0


def first_stage(candidate: dict) -> float:
    return reciprocal(candidate.get("sparse_rank")) + reciprocal(candidate.get("dense_rank"))


def fuse(candidates: list[dict], scores: dict[str, float], weight: float = 0.5) -> list[str]:
    retrieved = {candidate["table_id"]: first_stage(candidate)
                 for candidate in candidates if first_stage(candidate) > 0.0}
    ranked_by_model = sorted(
        (table_id for table_id in retrieved if table_id in scores),
        key=lambda table_id: (-scores[table_id], -retrieved[table_id]),
    )
    model_rank = {table_id: rank for rank, table_id in enumerate(ranked_by_model, 1)}
    return sorted(
        retrieved,
        key=lambda table_id: (
            -(weight * retrieved[table_id]
              + (1 - weight) * reciprocal(model_rank.get(table_id))),
            -retrieved[table_id],
        ),
    )


def load_scores(paths: list[Path]) -> dict[int, dict[str, float]]:
    # depth extensions overlap, so merge inside a question's dict too
    scored: dict[int, dict[str, float]] = defaultdict(dict)
    for path in paths:
        for record in load_jsonl(path):
            scored[record["id"]].update(record["scores"])
    return scored


def rankings(pairs: dict, scored: dict, weight: float) -> dict[str, list[str]]:
    ranking = {}
    for identifier, record in pairs.items():
        # without model scores, fall back to the retriever ranks rather than file order
        order = [table_id for table_id, _ in sorted(
            ((candidate["table_id"], first_stage(candidate))
             for candidate in record["candidates"] if first_stage(candidate) > 0.0),
            key=lambda pair: -pair[1],
        )]
        if identifier in scored:
            order = fuse(record["candidates"], scored[identifier], weight)
        ranking[str(identifier)] = order
    return ranking
