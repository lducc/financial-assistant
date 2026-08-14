#!/usr/bin/env python3
"""Export top-N candidate windows for listwise reranking.

The pointwise reranker judges each table alone, so two tables from the same
report that both carry the asked line item both score near 1.0 and nothing
compares them. Measured on the benchmark, the median score gap between a wrong
top-1 and the best gold is 0.067 and 47% of those misses are within 0.05. A
listwise pass shows the candidates side by side and asks for an order.

Windows are cut from an existing pairs export and an existing ranking rather than
by retrieving again: the candidate text is already materialized and the order is
the one the pointwise stage produced, so this is a reshaping step, not a
retrieval step.

Depth 20 is deliberate. Oracle reordering restricted to the head of the current
order reaches F2 0.7473 at top-10, 0.7893 at top-20 and 0.8021 at top-50 against
0.6562 today, so 20 captures 91% of what perfect reordering could ever buy while
keeping the window near 1,660 tokens.
"""

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def window(record: dict, order: list[str], depth: int, cap: int) -> list[dict]:
    """The first `depth` candidates in ranking order, each truncated to `cap`.

    Truncation is per candidate rather than over the whole window so that one
    verbose table cannot crowd the others out; the representation already puts
    the title, matched row and line-item inventory first, so the tail that goes
    is header and period boilerplate.
    """
    by_id = {candidate["table_id"]: candidate for candidate in record["candidates"]}
    ranked = [by_id[table_id] for table_id in order if table_id in by_id]
    # A ranking that does not cover the export falls back to sparse order rather
    # than silently emitting a short window.
    ranked.extend(candidate for candidate in record["candidates"] if candidate["table_id"] not in set(order))
    return [
        {"table_id": candidate["table_id"], "text": candidate["text"][:cap]}
        for candidate in ranked[:depth]
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--pairs", type=Path, required=True, help="an export from export_rerank_pairs.py")
    parser.add_argument("--ranking", type=Path, help="ranking.json to take the head from; sparse order if omitted")
    parser.add_argument("--output", type=Path, default=root / "output" / "rerank" / "windows.jsonl")
    parser.add_argument("--depth", type=int, default=20, help="candidates per window")
    parser.add_argument("--cap", type=int, default=250, help="characters kept per candidate")
    args = parser.parse_args()

    records = load_jsonl(args.pairs)
    ranking = json.loads(args.ranking.read_text(encoding="utf-8")) if args.ranking else {}
    rows, short = [], 0
    for record in records:
        order = ranking.get(str(record["id"]), [])
        if not order:
            order = [
                candidate["table_id"]
                for candidate in sorted(record["candidates"], key=lambda item: item["sparse_rank"])
            ]
        candidates = window(record, order, args.depth, args.cap)
        short += len(candidates) < args.depth
        rows.append({
            "id": record["id"],
            "question": record["question"],
            "line_items": record.get("line_items") or [],
            "candidates": candidates,
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    characters = sum(len(candidate["text"]) for row in rows for candidate in row["candidates"])
    print(json.dumps({
        "output": str(args.output),
        "questions": len(rows),
        "candidates": sum(len(row["candidates"]) for row in rows),
        "windows_below_depth": short,
        "mean_window_tokens": round(characters / max(1, len(rows)) / 3.5 + 220),
        "megabytes": round(args.output.stat().st_size / 1_048_576, 1),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
