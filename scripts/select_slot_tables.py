# Append high-confidence item-carrying tables to a starting ranking for complex questions.
#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path

from vifinqa.fusion import reciprocal
from vifinqa.jsonl import load_jsonl, write_jsonl
from vifinqa.slots import append_confidence_item_tables, load_line_items, named_line_items


def sparse_order(record: dict) -> list[str]:
    candidates = [
        candidate for candidate in record["candidates"]
        if reciprocal(candidate.get("sparse_rank")) > 0
    ]
    candidates.sort(key=lambda candidate: -reciprocal(candidate.get("sparse_rank")))
    return [candidate["table_id"] for candidate in candidates]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--base-ranking", type=Path, required=True)
    parser.add_argument("--starting-ranking", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--logit-gap", type=float, default=4.0)
    args = parser.parse_args()

    labels = load_line_items(root / "data" / "derived" / "line_items.json")
    records = sorted(load_jsonl(args.pairs), key=lambda record: record["id"])
    base = json.loads(args.base_ranking.read_text(encoding="utf-8"))
    starting = json.loads(args.starting_ranking.read_text(encoding="utf-8"))
    score_by_id = {record["id"]: record["scores"] for record in load_jsonl(args.scores)}

    rankings = {}
    traces = []
    for record in records:
        identifier = record["id"]
        order = base.get(str(identifier)) or sparse_order(record)
        start_order = starting.get(str(identifier), [])
        items = named_line_items(record["question"], labels)
        docs = record.get("selected_docs", [])
        years = set(re.findall(r"(?<!\d)20\d{2}(?!\d)", record["question"]))
        if len(docs) >= 2 or len(items) >= 2 or len(years) >= 2:
            selected = append_confidence_item_tables(
                record, start_order, order, items, score_by_id.get(identifier, {}),
                args.logit_gap,
            )
        else:
            selected = list(start_order)
        rankings[str(identifier)] = selected
        traces.append({
            "id": identifier,
            "line_items": items,
            "budget": min(30, max(1, 2 * len(docs))),
            "selected": selected,
            "slots": [],
            "covered_slots": 0,
            "input_count": len(order),
            "dropped": len(order) - len(selected),
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(rankings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    write_jsonl(args.trace, traces)
    print(json.dumps({
        "questions": len(records),
        "questions_with_line_items": sum(bool(trace["line_items"]) for trace in traces),
        "tables_kept": sum(len(trace["selected"]) for trace in traces),
        "questions_changed": sum(trace["selected"] != (base.get(str(trace["id"])) or []) for trace in traces),
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
