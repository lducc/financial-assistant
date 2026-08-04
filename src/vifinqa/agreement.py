"""E013 comparison utilities for two independently completed annotations."""

from __future__ import annotations

import json
import math


RELATIVE_TOLERANCE = 0.0002  # organizer clarification: 0.02%


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _numeric_agreement(left: object, right: object) -> tuple[bool, float | None]:
    try:
        left_value, right_value = float(left), float(right)
    except (TypeError, ValueError):
        return False, None
    if not math.isfinite(left_value) or not math.isfinite(right_value):
        return False, None
    relative_difference = abs(left_value - right_value) / max(abs(left_value), abs(right_value), 1e-12)
    return relative_difference <= RELATIVE_TOLERANCE, relative_difference


def compare_pair(left: dict, right: dict) -> dict:
    left_annotation, right_annotation = left["annotation"], right["annotation"]
    answer_agree, relative_difference = _numeric_agreement(left_annotation.get("numeric_answer"), right_annotation.get("numeric_answer"))
    return {
        "id": left["id"],
        "report_set_exact": set(left_annotation.get("gold_reports") or []) == set(right_annotation.get("gold_reports") or []),
        "table_set_exact": set(left_annotation.get("gold_tables") or []) == set(right_annotation.get("gold_tables") or []),
        "operation_graph_exact": _canonical(left_annotation.get("operation_graph")) == _canonical(right_annotation.get("operation_graph")),
        "unit_exact": left_annotation.get("table_units") == right_annotation.get("table_units"),
        "numeric_within_0_02_percent": answer_agree,
        "numeric_relative_difference": relative_difference,
    }


def compare_reviews(left_records: list[dict], right_records: list[dict]) -> dict:
    left_by_id = {record["id"]: record for record in left_records}
    right_by_id = {record["id"]: record for record in right_records}
    common = sorted(set(left_by_id) & set(right_by_id))
    rows = [compare_pair(left_by_id[item], right_by_id[item]) for item in common]
    measures = ("report_set_exact", "table_set_exact", "operation_graph_exact", "unit_exact", "numeric_within_0_02_percent")
    return {
        "common_ids": common,
        "left_only_ids": sorted(set(left_by_id) - set(right_by_id)),
        "right_only_ids": sorted(set(right_by_id) - set(left_by_id)),
        "pairs": rows,
        "agreement": {measure: sum(bool(row[measure]) for row in rows) for measure in measures},
        "pair_count": len(rows),
        "disagreements": [row for row in rows if not all(row[measure] for measure in measures)],
        "numeric_tolerance": RELATIVE_TOLERANCE,
        "note": "Numeric agreement is symmetric across reviewers; organizer scoring is relative to hidden gold.",
    }
