#!/usr/bin/env python3
"""Build the line-addressed ViFinQA table catalog required for submissions."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifinqa.catalog import iter_report_paths, parse_report_identity
from vifinqa.line_catalog import extract_submission_table_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "derived" / "table_catalog")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    catalog_path = args.output_dir / "tables.jsonl"
    count = 0
    report_count = 0
    line_ids: set[str] = set()
    scopes: Counter[str] = Counter()
    with catalog_path.open("w", encoding="utf-8") as output:
        for path in iter_report_paths(args.dataset_root):
            identity = parse_report_identity(path, args.dataset_root)
            text = path.read_text(encoding="utf-8")
            records = extract_submission_table_records(text, identity)
            report_count += 1
            scopes[identity.scope] += 1
            for record in records:
                if record.submission_table_id in line_ids:
                    raise ValueError(f"Duplicate submission table ID: {record.submission_table_id}")
                line_ids.add(record.submission_table_id)
                output.write(json.dumps(record.as_dict(), ensure_ascii=False) + "\n")
                count += 1
    summary = {
        "catalog_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "identifier_semantics": "report_id|1-based line at literal <table> start in source OCR text",
        "report_count": report_count,
        "table_count": count,
        "unique_submission_table_ids": len(line_ids),
        "scope_counts": dict(sorted(scopes.items())),
        "catalog_path": str(catalog_path),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

