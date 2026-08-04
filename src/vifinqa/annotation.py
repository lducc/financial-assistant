"""E010 validation rules for manually completed E003 proxy labels."""

from __future__ import annotations

import math
import re


TABLE_ID = re.compile(r"^([^|\s]+)\|([1-9]\d*)$")
COMPLETE_FIELDS = (
    "annotator", "question_slots", "required_metric_roles", "gold_reports",
    "gold_tables", "row_column_bindings", "table_units", "operation_graph",
    "pandas_query", "numeric_answer", "confidence",
)


def _present(value: object) -> bool:
    return value is not None and value != "" and value != [] and value != {}


def validate_record(record: dict, allow_incomplete: bool = False) -> list[str]:
    annotation = record.get("annotation")
    if not isinstance(annotation, dict):
        return ["annotation must be an object"]
    status = annotation.get("status")
    if status == "unannotated":
        return [] if allow_incomplete else ["unannotated record is not scoreable"]
    if status not in {"complete", "needs_review", "excluded"}:
        return ["status must be unannotated, complete, needs_review, or excluded"]
    if status == "excluded":
        return [] if _present(annotation.get("failure_tags")) else ["excluded record needs failure_tags"]
    errors = [f"missing {field}" for field in COMPLETE_FIELDS if not _present(annotation.get(field))]
    answer = annotation.get("numeric_answer")
    if _present(answer):
        try:
            if not math.isfinite(float(answer)):
                errors.append("numeric_answer must be finite")
        except (TypeError, ValueError):
            errors.append("numeric_answer must be numeric")
    reports = annotation.get("gold_reports") or []
    if not isinstance(reports, list) or not all(isinstance(item, str) and item for item in reports):
        errors.append("gold_reports must be a non-empty string list")
        reports = []
    tables = annotation.get("gold_tables") or []
    if not isinstance(tables, list) or not tables:
        errors.append("gold_tables must be a non-empty list")
    else:
        for table_id in tables:
            match = TABLE_ID.match(str(table_id))
            if not match:
                errors.append(f"invalid table id: {table_id}")
            elif match.group(1) not in reports:
                errors.append(f"table report missing from gold_reports: {table_id}")
    if _present(annotation.get("pandas_query")) and not str(annotation["pandas_query"]).lstrip().startswith("result"):
        errors.append("pandas_query must assign result")
    confidence = annotation.get("confidence")
    if _present(confidence) and confidence not in {"high", "medium", "low"}:
        errors.append("confidence must be high, medium, or low")
    return errors
