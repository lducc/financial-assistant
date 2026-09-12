import argparse
import json
from pathlib import Path

from vifinqa.fusion import load_scores, rankings
from vifinqa.jsonl import load_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--pairs", type=Path, default=root / "output" / "rerank" / "pairs.jsonl")
    parser.add_argument("--scores", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, default=root / "output" / "rerank" / "ranking.json")
    parser.add_argument("--weight", type=float, default=0.5,
                        help="share of the reciprocal-rank sum given to the sparse rank")
    parser.add_argument("--first-stage-only", action="store_true",
                        help="rank by the retrievers alone, with no reranker scores")
    args = parser.parse_args()

    pairs = {record["id"]: record for record in load_jsonl(args.pairs)}
    if not args.scores and not args.first_stage_only:
        raise SystemExit("pass --scores, or --first-stage-only to rank without a model")
    scored = load_scores(args.scores)
    ranking = rankings(pairs, scored, args.weight)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(ranking, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "questions": len(ranking),
        "scored": len(scored),
        "unscored_kept_as_is": len(set(pairs) - set(scored)),
    }, indent=2))


if __name__ == "__main__":
    main()
