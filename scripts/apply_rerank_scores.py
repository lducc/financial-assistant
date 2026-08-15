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
import json
from pathlib import Path

from vifinqa.fusion import load_scores, rankings
from vifinqa.jsonl import load_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--pairs", type=Path, default=root / "output" / "rerank" / "pairs.jsonl")
    parser.add_argument(
        "--scores", type=Path, action="append", default=[],
        help="score file from a Kaggle run; repeat once per shard (defaults to output/rerank/scores.jsonl)",
    )
    parser.add_argument("--output", type=Path, default=root / "output" / "rerank" / "ranking.json")
    parser.add_argument("--mode", choices=("fuse", "replace"), default="fuse")
    parser.add_argument(
        "--replace-tiers", default="",
        help="SUPERSEDED by --weight; kept to rebuild the Tables F2 0.5221 submission. A step "
             "function on the tier classifier, whose subset was chosen by eye from sixteen. "
             "Under nested CV the folds pick a different subset four times in five and it gains "
             "+0.0019 out-of-fold, against +0.0066 for a single weight picked in four folds of five",
    )
    parser.add_argument("--tiers", type=Path, default=root / "data" / "derived" / "question_tiers.jsonl")
    parser.add_argument(
        "--weight", type=float, default=0.5,
        help="share of the reciprocal-rank sum given to the sparse rank; 0.5 is the shipped "
             "unweighted fusion and nested CV picks 0.3, a difference inside the noise floor",
    )
    args = parser.parse_args()

    pairs = {record["id"]: record for record in load_jsonl(args.pairs)}
    scored = load_scores(args.scores or [args.pairs.parent / "scores.jsonl"])
    replace_tiers = {tier.strip() for tier in args.replace_tiers.split(",") if tier.strip()}
    tier_of = {record["id"]: record["tier"] for record in load_jsonl(args.tiers)} if replace_tiers else {}
    ranking = rankings(pairs, scored, args.mode, args.weight, replace_tiers, tier_of)

    replaced = sum(1 for i in pairs if i in scored and tier_of.get(i) in replace_tiers)
    moved = sum(
        1 for identifier, record in pairs.items()
        if ranking[str(identifier)][0] != record["candidates"][0]["table_id"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(ranking, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "mode": args.mode,
        "replace_tiers": sorted(replace_tiers),
        "weight": args.weight,
        "questions": len(ranking),
        "scored": len(scored),
        "unscored_kept_as_is": len(set(pairs) - set(scored)),
        "questions_replaced_by_tier": replaced,
        "top_table_changed": moved,
        "top_table_unchanged": len(ranking) - moved,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
