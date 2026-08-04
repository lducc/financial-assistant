#!/usr/bin/env python3
"""Validate E003 annotation JSONL without scoring it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifinqa.annotation import validate_record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    invalid = []
    total = 0
    statuses: dict[str, int] = {}
    for line_number, line in enumerate(args.input.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        total += 1
        record = json.loads(line)
        status = record.get("annotation", {}).get("status", "missing")
        statuses[status] = statuses.get(status, 0) + 1
        errors = validate_record(record, allow_incomplete=args.allow_incomplete)
        if errors:
            invalid.append({"line": line_number, "id": record.get("id"), "errors": errors})
    summary = {"records": total, "statuses": statuses, "invalid": invalid, "valid": total - len(invalid)}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    raise SystemExit(1 if invalid else 0)


if __name__ == "__main__":
    main()
