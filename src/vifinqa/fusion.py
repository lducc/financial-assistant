"""Turn cross-encoder scores into the ordering that ships.

Every replacement tried on this task has lost and every fusion that earned its
place has won, so the reranker enters as one more reciprocal rank rather than as
the new order. That also bounds the damage if the model is wrong: at worst it
pulls the ranking halfway toward its own opinion, instead of replacing ours.

This lives in the package rather than beside the CLI that writes ranking.json
because `compare_rerank_runs.py` has to score exactly what ships. Two copies of
the rule means a comparison can silently stop measuring the shipped system.
"""

from collections import defaultdict
from pathlib import Path

from vifinqa.jsonl import load_jsonl


RRF_OFFSET = 60




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
