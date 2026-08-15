#!/usr/bin/env python3
"""Build a hard-and-intermediate benchmark from reviewed proposals.

Why a second gold set at all. On the mixed benchmark only 36 of 233 questions
changed under the last accepted ranking change and 18 under the one before, so
the effective sample is a fraction of the label count: easy questions sit at F2
0.75 and do not move. Measured on the one large effect we have — the reranker
against sparse — signal-to-noise per label is 1.45x higher on hard questions than
on the mixed set, so the same interval costs 0.48x the labels. A 200-question
hard set is worth roughly 415 mixed ones for deciding ranking quality, which is
the only axis with headroom left.

Why the narrowing rule is the whole design. `accept_proposals.py` binds every
table whose row matches the line item. On the easy-dominated `train/accepted.jsonl`
that gives 3.63 gold tables per question against the organizers' 3.29 — already
wide enough that the v3 and v4 rankings tie there at +0.0008 while live moved
+0.0258. Over-wide gold counts every restatement as correct, so no ranking can be
told from another. On hard questions the same gate gives a median of 18 tables,
which would be useless. 69.1% of (item, report) pairs offer more than one table.

So exactly one table is bound per (item, report): the question asks for one
figure, and one table reports it. Restatements are kept on the record as
`restatements` but left unbound, which reproduces the structure that makes
`annotations/benchmark.jsonl` a working ruler — `gold_tables_for(..., "binding")`
narrows it 4.50 -> 3.24, matching the organizers' 3.29 implied by live precision
and recall.

The choice among candidates is fixed in advance and reads nothing from any score:

1. rows describing a movement rather than a balance are dropped, as in `accept_proposals.py`
2. a primary statement beats a note, because a note restates what the statement reports
3. within a class, the row label that matches the item most tightly wins
4. ties break on the lowest line number, so the rule is deterministic

Questions whose branch the proposer could not resolve — "the year revenue grew
fastest" needs every year to choose and one to answer — are not guessed. They are
written to the deferral file with a reason, following the v3 protocol.
"""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from accept_proposals import is_movement
from propose_multihop_labels import named_line_items
from vifinqa.answers import fold

# A statement reports the figure; a note restates it. Matched on the folded title
# because OCR mangles diacritics and case inconsistently.
STATEMENT_MARKERS = (
    "bang can doi ke toan", "bao cao ket qua hoat dong kinh doanh",
    "bao cao luu chuyen tien te", "bao cao tinh hinh tai chinh",
)
NOTE_MARKERS = ("thuyet minh",)


def title_class(title: str) -> int:
    """0 for a primary statement, 1 for a note, 2 for anything unrecognised."""
    folded = fold(title or "")
    if any(marker in folded for marker in STATEMENT_MARKERS):
        return 0
    if any(marker in folded for marker in NOTE_MARKERS):
        return 1
    return 2


def line_number(table_id: str) -> int:
    tail = table_id.rpartition("|")[2]
    return int(tail) if tail.isdigit() else 0


def choose(hits: list[dict], item: str) -> dict | None:
    """The one table that reports this item in this report, or None.

    Tightness of the label match is length: a row called exactly the item is a
    better binding than one that contains it inside a longer phrase, which is
    usually a sub-line or a differently-scoped variant of the same account.
    """
    usable = [hit for hit in hits if not is_movement(hit["row_label"], item)]
    if not usable:
        return None
    return min(
        usable,
        key=lambda hit: (
            title_class(hit.get("title", "")),
            len(fold(hit["row_label"])),
            line_number(hit["table"]),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposals", type=Path, default=ROOT / "annotations" / "train" / "proposals.jsonl")
    parser.add_argument("--tiers", type=Path, default=ROOT / "data" / "derived" / "question_tiers.jsonl")
    parser.add_argument(
        "--exclude", type=Path, action="append", default=[],
        help="label file whose questions must stay out; repeat per file",
    )
    parser.add_argument("--tier", action="append", default=[], help="tiers to keep (default hard and intermediate)")
    parser.add_argument("--output", type=Path, default=ROOT / "annotations" / "benchmark_hard.jsonl")
    parser.add_argument("--deferred", type=Path, default=ROOT / "annotations" / "benchmark_hard_deferred.jsonl")
    parser.add_argument(
        "--max-gated-reports", type=int, default=3,
        help="reject a question gating more reports than this. Capping the bound table count "
             "alone does not work — the cap and the resulting width move together, so it only "
             "selects questions that happen to fit. Gating few reports is the property that "
             "makes a question determinate by reference: the proposer treats every gated report "
             "as evidence-bearing, which is right when the question names its years and entities "
             "and wrong when an intermediate result has to choose them. This keeps the first kind.",
    )
    parser.add_argument(
        "--max-tables", type=int, default=12,
        help="reject a question that still binds more than this after narrowing; a question "
             "needing more evidence than the whole submission budget cannot score anything useful",
    )
    args = parser.parse_args()

    def load(path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    keep_tiers = set(args.tier) or {"hard", "intermediate"}
    excluded = {record["id"] for path in args.exclude for record in load(path)}
    tiers = {record["id"]: record["tier"] for record in load(args.tiers)}

    accepted, deferred, reasons = [], [], Counter()
    widths = []
    for record in load(args.proposals):
        identifier = record["id"]
        if identifier in excluded or tiers.get(identifier) not in keep_tiers:
            continue
        if record.get("status") != "proposed":
            reasons[record["status"]] += 1
            deferred.append({"id": identifier, "reason": record.get("status")})
            continue
        # An item the generic filter dropped means the recorded evidence is only
        # part of what the question needs, which would score as if it were all.
        if len(record["line_items"]) != len(named_line_items(record["question"])):
            reasons["item_dropped_as_generic"] += 1
            deferred.append({"id": identifier, "reason": "item_dropped_as_generic"})
            continue
        gated = len(record.get("gated_reports") or [])
        if gated > args.max_gated_reports:
            reasons["branch_unresolved"] += 1
            deferred.append({
                "id": identifier, "reason": "branch_unresolved", "gated_reports": gated,
                "note": "the question selects among these reports; which ones it needs is not "
                        "derivable from the row-label search and is not guessed",
            })
            continue

        bindings, restatements = [], set()
        for item, per_report in (record.get("evidence") or {}).items():
            for report, hits in per_report.items():
                best = choose(hits, item)
                if best is None:
                    continue
                column = best["columns"][0]
                bindings.append({
                    "role": item.replace(" ", "_"),
                    "table": best["table"],
                    "row": best["row"],
                    "column": column["column"],
                    "row_label": best["row_label"],
                    "raw": column["raw"],
                })
                restatements.update(
                    hit["table"] for hit in hits if hit["table"] != best["table"]
                )
        if not bindings:
            reasons["all_rows_were_movements"] += 1
            deferred.append({"id": identifier, "reason": "all_rows_were_movements"})
            continue
        tables = sorted({binding["table"] for binding in bindings})
        if len(tables) > args.max_tables:
            reasons["too_many_tables_after_narrowing"] += 1
            deferred.append({
                "id": identifier, "reason": "too_many_tables_after_narrowing",
                "tables": len(tables), "gated_reports": len(record.get("gated_reports") or []),
            })
            continue

        widths.append(len(tables))
        accepted.append({
            "id": identifier,
            "question": record["question"],
            "tier": record["tier"],
            "source": "hard_v1",
            "weight": 1.0,
            "taxonomy": {"operation": "unlabelled", "table_count": len(tables)},
            "provenance": {
                "discovery": "folded row-label search over raw OCR of gated reports",
                "proposer": "scripts/propose_multihop_labels.py",
                "narrowing": "one binding per (line item, report); statement over note; "
                             "tightest label; lowest line number",
                "line_items": record["line_items"],
                "independent_of_retriever": True,
                "batch": "hard_v1",
            },
            "annotation": {
                "status": "complete",
                "gold_reports": sorted({table.partition("|")[0] for table in tables}),
                # Wide keeps the restatements so the two gold definitions differ
                # here the way they do on the mixed benchmark; binding narrows.
                "gold_tables": sorted(set(tables) | restatements),
                "row_column_bindings": bindings,
            },
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in accepted), encoding="utf-8"
    )
    args.deferred.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in deferred), encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output),
        "deferred": str(args.deferred),
        "accepted": len(accepted),
        "accepted_by_tier": dict(Counter(record["tier"] for record in accepted)),
        "deferred_count": len(deferred),
        "deferral_reasons": dict(reasons),
        "bound_tables_per_question": round(sum(widths) / max(1, len(widths)), 2),
        "wide_tables_per_question": round(sum(
            len(record["annotation"]["gold_tables"]) for record in accepted
        ) / max(1, len(accepted)), 2),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
