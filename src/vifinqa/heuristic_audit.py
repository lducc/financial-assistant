"""E014 compare two non-gold operation heuristics without selecting a winner."""

from __future__ import annotations

from collections import Counter


E003_TO_E009 = {"extremum_or_selector": "selector"}


def compare_operation_hints(queue: list[dict], plans: list[dict]) -> dict:
    by_id = {plan["id"]: plan for plan in plans}
    rows = []
    for record in queue:
        identifier = record["id"]
        if identifier not in by_id:
            raise ValueError(f"Missing E009 plan for E003 ID {identifier}")
        original = record["features"]["operation_hint"]
        mapped = E003_TO_E009.get(original, original)
        planned = by_id[identifier]["operation"]
        rows.append({"id": identifier, "e003_hint": original, "e003_mapped": mapped, "e009_plan": planned, "agrees": mapped == planned})
    if len(by_id) != len(plans):
        raise ValueError("Duplicate E009 plan IDs")
    matrix = Counter((row["e003_mapped"], row["e009_plan"]) for row in rows)
    return {
        "records": len(rows),
        "agree_count": sum(row["agrees"] for row in rows),
        "disagree_count": sum(not row["agrees"] for row in rows),
        "matrix": {f"{left} -> {right}": count for (left, right), count in sorted(matrix.items())},
        "disagreements": [row for row in rows if not row["agrees"]],
        "note": "Heuristic agreement is not semantic accuracy; retain disagreements for human annotation.",
    }
