#!/usr/bin/env python3
"""Re-derive every benchmark claim from raw OCR and fail loudly on any mismatch.

A label set is only worth what can be rechecked. This walks the whole benchmark
and confirms, per record: the schema is complete, the table IDs parse and exist,
every bound cell still holds the exact raw string recorded, the gold tables and
gold reports agree, and the corpus is the one the manifest was built against.

Exit status is non-zero when anything fails, so it can gate a change.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

from vifinqa.evaluation_v2 import corpus_tree_hash, sha256_text
from vifinqa.jsonl import load_jsonl
from vifinqa.review import source_report_catalog, validate_source_bindings

ROOT = Path(__file__).resolve().parents[1]

REQUIRED = ("id", "question", "tier", "source", "taxonomy", "provenance", "annotation")
TIERS = {"easy", "medium", "intermediate", "hard", "unclassified"}


def check_record(record: dict, reports: dict) -> list[str]:
    errors = [f"missing field: {field}" for field in REQUIRED if field not in record]
    if errors:
        return errors
    if record["tier"] not in TIERS:
        errors.append(f"unknown tier: {record['tier']}")
    annotation = record["annotation"]
    tables, gold_reports = annotation["gold_tables"], annotation["gold_reports"]
    if not tables:
        errors.append("no gold tables")
    if sorted({table.partition("|")[0] for table in tables}) != sorted(gold_reports):
        errors.append("gold_reports do not match the reports named by gold_tables")
    for report in gold_reports:
        if report not in reports:
            errors.append(f"gold report missing from corpus: {report}")
    bound = {binding["table"] for binding in annotation["row_column_bindings"]}
    if bound - set(tables):
        errors.append(f"bindings reference tables outside gold_tables: {sorted(bound - set(tables))}")
    if record["taxonomy"]["table_count"] != len(tables):
        errors.append("taxonomy.table_count disagrees with gold_tables")
    # The expensive check: every bound cell must still hold its recorded raw string.
    errors.extend(validate_source_bindings(record, ROOT / "data" / "raw" / "vifinqa", reports))
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data" / "raw" / "vifinqa")
    parser.add_argument("--benchmark", type=Path, default=ROOT / "annotations" / "benchmark.jsonl")
    parser.add_argument("--manifest", type=Path, default=ROOT / "annotations" / "benchmark_manifest.json")
    parser.add_argument("--skip-corpus-hash", action="store_true", help="skip the slow corpus tree hash")
    args = parser.parse_args()

    records = load_jsonl(args.benchmark)
    reports = source_report_catalog(args.dataset_root)

    failures: dict[int, list[str]] = {}
    identifiers = Counter(record["id"] for record in records)
    duplicates = [identifier for identifier, count in identifiers.items() if count > 1]
    for record in records:
        errors = check_record(record, reports)
        if errors:
            failures[record["id"]] = errors

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    drift = []
    if manifest["records"] != len(records):
        drift.append(f"manifest records={manifest['records']} file={len(records)}")
    if manifest["benchmark_sha256"] != sha256_text(args.benchmark.read_text(encoding="utf-8")):
        drift.append("benchmark_sha256 does not match the file; rebuild the manifest")
    if not args.skip_corpus_hash and manifest["corpus_tree_hash"] != corpus_tree_hash(args.dataset_root):
        drift.append("corpus_tree_hash does not match this corpus; results are not comparable")

    bindings = sum(len(record["annotation"]["row_column_bindings"]) for record in records)
    report = {
        "benchmark": str(args.benchmark),
        "records": len(records),
        "verified_bindings": bindings,
        "records_without_bindings": sum(
            1 for record in records if not record["annotation"]["row_column_bindings"]
        ),
        "duplicate_ids": duplicates,
        "manifest_drift": drift,
        "failures": {str(identifier): errors for identifier, errors in sorted(failures.items())},
        "status": "VALID" if not (failures or duplicates or drift) else "INVALID",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "VALID":
        sys.exit(1)


if __name__ == "__main__":
    main()
