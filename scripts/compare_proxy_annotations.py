#!/usr/bin/env python3
"""Compare two E012 reviewer files and write a reconciliation report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifinqa.agreement import compare_reviews
from vifinqa.annotation import validate_record


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    left, right = read_jsonl(args.left), read_jsonl(args.right)
    invalid = {"left": [record["id"] for record in left if validate_record(record)], "right": [record["id"] for record in right if validate_record(record)]}
    if invalid["left"] or invalid["right"]:
        raise SystemExit(f"Refusing comparison of invalid completed labels: {invalid}")
    report = compare_reviews(left, right)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("pair_count", "agreement", "left_only_ids", "right_only_ids")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
