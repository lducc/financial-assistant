"""Check independent v2 reviewer calibration before adjudication starts."""

import argparse
import json
import statistics
from pathlib import Path

from vifinqa.evaluation_v2 import (
    _required_tables,
    index_records,
    load_jsonl,
    validate_v2_annotation,
    validate_v2_source_bindings,
)

ROOT = Path(__file__).resolve().parents[1]


def slot_signature(slot: dict) -> tuple:
    alternatives = tuple(sorted(tuple(sorted(_required_tables(alternative))) for alternative in slot["alternatives"]))
    return (
        slot["entity"], slot["report_year"], slot["scope"], slot["metric"],
        slot["operand_role"], alternatives,
    )


def table_set(record: dict) -> set[str]:
    return {
        table
        for slot in record["slots"]
        for alternative in slot["alternatives"]
        for table in _required_tables(alternative)
    }


def jaccard(left: set[str], right: set[str]) -> float:
    return len(left & right) / len(left | right) if left or right else 1.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviewer-a", type=Path, required=True)
    parser.add_argument("--reviewer-b", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--expected-records", type=int, default=20)
    args = parser.parse_args()
    reviewer_a = index_records(load_jsonl(args.reviewer_a), "reviewer A")
    reviewer_b = index_records(load_jsonl(args.reviewer_b), "reviewer B")
    if len(reviewer_a) != args.expected_records or set(reviewer_a) != set(reviewer_b):
        raise ValueError("reviewer files must have exact matching calibration IDs")
    coordinate_errors = []
    for name, records in (("reviewer_a", reviewer_a), ("reviewer_b", reviewer_b)):
        for identifier, record in records.items():
            coordinate_errors.extend(f"{name} id={identifier}: {error}" for error in validate_v2_annotation(record))
            coordinate_errors.extend(
                f"{name} id={identifier}: {error}"
                for error in validate_v2_source_bindings(record, args.raw_root)
            )
    if coordinate_errors:
        raise ValueError("calibration validation failed:\n" + "\n".join(coordinate_errors[:20]))
    if any(reviewer_a[identifier]["reviewer_protocol_hash"] == reviewer_b[identifier]["reviewer_protocol_hash"] for identifier in reviewer_a):
        raise ValueError("reviewer protocol hashes must differ")
    table_jaccards, slot_agreements, operation_agreements = [], [], []
    for identifier in sorted(reviewer_a):
        left, right = reviewer_a[identifier], reviewer_b[identifier]
        table_jaccards.append(jaccard(table_set(left), table_set(right)))
        left_slots = set(map(slot_signature, left["slots"]))
        right_slots = set(map(slot_signature, right["slots"]))
        slot_agreements.append(len(left_slots & right_slots) / max(len(left_slots), len(right_slots)))
        operation_agreements.append(left["operation"] == right["operation"])
    result = {
        "records": len(reviewer_a),
        "source_coordinate_valid": True,
        "median_table_set_jaccard": statistics.median(table_jaccards),
        "slot_exact_agreement": sum(slot_agreements) / len(slot_agreements),
        "operation_agreement": sum(operation_agreements) / len(operation_agreements),
    }
    result["passed"] = (
        result["median_table_set_jaccard"] >= 0.80
        and result["slot_exact_agreement"] >= 0.75
        and result["operation_agreement"] >= 0.80
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
