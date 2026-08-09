#!/usr/bin/env python3
"""Validate the independent pilot's source-cell bindings."""

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifinqa.review import source_report_catalog, validate_source_bindings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data" / "raw" / "vifinqa")
    parser.add_argument("--labels", type=Path, default=ROOT / "annotations" / "pilot_v1" / "agent_labels.jsonl")
    args = parser.parse_args()
    reports = source_report_catalog(args.dataset_root)
    errors = []
    for line in args.labels.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            errors.extend(f"id={record.get('id')}: {error}" for error in validate_source_bindings(record, args.dataset_root, reports))
    if errors:
        raise SystemExit("\n".join(errors))
    print("VALID")


if __name__ == "__main__":
    main()
