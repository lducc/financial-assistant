#!/usr/bin/env python3
"""Fuse cross-encoder scores from Kaggle back into the sparse ranking.

Every replacement tried on this task has lost and every fusion that earned its
place has won, so the reranker enters as one more reciprocal rank rather than as
the new order. That also bounds the damage if the model is wrong: at worst it
pulls the ranking halfway toward its own opinion, instead of replacing ours.

Writes reranked orderings keyed by question ID, which the evaluator and the
submission builder both read.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path


RRF_OFFSET = 60


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def fuse(candidates: list[dict], scores: dict[str, float], mode: str) -> list[str]:
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
                1 / (RRF_OFFSET + sparse[table_id])
                + (1 / (RRF_OFFSET + dense_rank[table_id]) if table_id in dense_rank else 0.0)
            ),
            sparse[table_id],
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--pairs", type=Path, default=root / "output" / "rerank" / "pairs.jsonl")
    parser.add_argument("--scores", type=Path, default=root / "output" / "rerank" / "scores.jsonl")
    parser.add_argument("--output", type=Path, default=root / "output" / "rerank" / "ranking.json")
    parser.add_argument("--mode", choices=("fuse", "replace"), default="fuse")
    args = parser.parse_args()

    pairs = {record["id"]: record for record in load_jsonl(args.pairs)}
    scored = {record["id"]: record["scores"] for record in load_jsonl(args.scores)}
    missing = sorted(set(pairs) - set(scored))

    ranking = {}
    moved = defaultdict(int)
    for identifier, record in pairs.items():
        order = [candidate["table_id"] for candidate in record["candidates"]]
        if identifier in scored:
            fused = fuse(record["candidates"], scored[identifier], args.mode)
            moved[fused[0] != order[0]] += 1
            ranking[str(identifier)] = fused
        else:
            ranking[str(identifier)] = order

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(ranking, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "mode": args.mode,
        "questions": len(ranking),
        "scored": len(scored),
        "unscored_kept_as_is": len(missing),
        "top_table_changed": moved[True],
        "top_table_unchanged": moved[False],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
