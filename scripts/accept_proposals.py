#!/usr/bin/env python3
"""Turn reviewed proposals into benchmark records, conservatively.

`propose_multihop_labels.py` does the mechanical search; this applies the rules a
reviewer would and refuses anything they would have to think about:

* A question is accepted only if every line item it names survived the generic
  filter. Q400 asks for cash flow in the year revenue grew fastest, and "doanh thu
  thuần" matches too many rows to survive; labelling it from the cash-flow item
  alone would record half the evidence as if it were all of it.
* Rows whose label merely contains the item are dropped when a movement verb sits
  in front of it: "Chuyển sang chi phí trả trước dài hạn" is a transfer into the
  account, not the balance the question asks for.
* Every accepted binding is written with the raw cell string read back from
  source, so `scripts/verify_benchmark.py` can re-derive it.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from propose_multihop_labels import named_line_items
from vifinqa.answers import fold
from vifinqa.jsonl import load_jsonl

# Verbs that turn a balance into a movement, a transfer, or an allocation.
MOVEMENT_PREFIXES = (
    "chuyen sang", "chuyen tu", "tang", "giam", "phan bo", "trich", "hoan nhap",
    "so da", "da tra", "da thu", "thanh ly", "mua sam",
)


def is_movement(row_label: str, item: str) -> bool:
    """Whether the row describes a change in the account rather than its balance."""
    folded = fold(row_label)
    before = folded.split(item)[0] if item in folded else ""
    return any(prefix in before for prefix in MOVEMENT_PREFIXES)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposals", type=Path, default=ROOT / "annotations" / "v3" / "proposals.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "annotations" / "v3" / "labels_proposed.jsonl")
    parser.add_argument("--max-tables", type=int, default=12, help="reject proposals larger than this")
    args = parser.parse_args()

    proposals = load_jsonl(args.proposals)
    accepted, rejected = [], Counter()
    for record in proposals:
        if record["status"] != "proposed":
            rejected[record["status"]] += 1
            continue
        # Items the question names, before the generic filter removed any.
        if len(record["line_items"]) != len(named_line_items(record["question"])):
            rejected["item_dropped_as_generic"] += 1
            continue

        bindings = []
        for item, per_report in record["evidence"].items():
            for hits in per_report.values():
                for hit in hits:
                    if is_movement(hit["row_label"], item):
                        continue
                    column = hit["columns"][0]
                    bindings.append({
                        "role": item.replace(" ", "_"),
                        "table": hit["table"],
                        "row": hit["row"],
                        "column": column["column"],
                        "row_label": hit["row_label"],
                        "raw": column["raw"],
                    })
        if not bindings:
            rejected["all_rows_were_movements"] += 1
            continue
        tables = sorted({binding["table"] for binding in bindings})
        if len(tables) > args.max_tables:
            rejected["too_many_tables"] += 1
            continue

        accepted.append({
            "id": record["id"],
            "question": record["question"],
            "tier": record["tier"],
            "taxonomy": {"operation": "unlabelled", "table_count": len(tables)},
            "provenance": {
                "discovery": "folded row-label search over raw OCR of gated reports",
                "proposer": "scripts/propose_multihop_labels.py",
                "line_items": record["line_items"],
                "reviewer": "claude_proposal_review_2026-08-13",
                "independent_of_retriever": True,
                "batch": "v3_proposed",
            },
            "annotation": {
                "status": "complete",
                "gold_reports": sorted({table.partition("|")[0] for table in tables}),
                "gold_tables": tables,
                "row_column_bindings": bindings,
            },
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in accepted), encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output),
        "proposals": len(proposals),
        "accepted": len(accepted),
        "accepted_by_tier": dict(Counter(record["tier"] for record in accepted)),
        "rejected": dict(rejected),
        "mean_tables": round(
            sum(len(record["annotation"]["gold_tables"]) for record in accepted) / max(1, len(accepted)), 2
        ),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
