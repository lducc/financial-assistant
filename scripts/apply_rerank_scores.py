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


def fuse(
    candidates: list[dict], scores: dict[str, float], mode: str, weight: float = 0.5,
) -> list[str]:
    """Order one question's candidates by weighted reciprocal rank.

    `weight` is the share given to the sparse rank: 1.0 is sparse alone, 0.0 is
    the model alone, 0.5 is the unweighted sum that every submission through
    Tables F2 0.5221 used.

    Read the sweep before turning this knob. Nested five-fold cross-validation
    over the benchmark picks w=0.3 in four folds of five and is worth +0.0066
    out-of-fold, CI [-0.0191, +0.0306]. The in-sample curve is jagged — 0.3 scores
    0.6714 and 0.4 scores 0.6546 — and the five folds range 0.5890 to 0.6859 on 45
    questions each. Both say the instrument resolves a paired delta to about
    ±0.02, and every weight in 0.0-0.5 sits inside that. Nothing here is
    distinguishable on 233 records; treat a change as free rather than as a gain.
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

    Sharded Kaggle runs write one file per GPU and never share a question, so a
    union over questions was enough. A depth extension does share questions and
    differs only in which candidates it judged, so the union has to reach inside;
    on disjoint shards that is the same thing.
    """
    scored: dict[int, dict[str, float]] = defaultdict(dict)
    for path in paths:
        for record in load_jsonl(path):
            scored[record["id"]].update(record["scores"])
    return scored


def rankings(
    pairs: dict, scored: dict, mode: str, weight: float, replace_tiers: set, tier_of: dict,
) -> dict[str, list[str]]:
    """The shipped ordering rule, keyed by question ID as ranking.json is.

    Lives here rather than in each caller so that a comparison against the
    shipped ranking cannot drift away from what actually ships.
    """
    ranking = {}
    for identifier, record in pairs.items():
        order = [candidate["table_id"] for candidate in record["candidates"]]
        if identifier in scored:
            # A question with no tier keeps the default; the classifier covers all
            # 1,012, so this only fires on label sets it has not been run over.
            question_mode = "replace" if tier_of.get(identifier) in replace_tiers else mode
            order = fuse(record["candidates"], scored[identifier], question_mode, weight)
        ranking[str(identifier)] = order
    return ranking


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
