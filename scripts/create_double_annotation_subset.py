#!/usr/bin/env python3
"""Create the immutable E012 reviewer subset and its reproducibility summary."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifinqa.double_annotation import select_double_annotation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--size", type=int, default=30)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.queue.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = select_double_annotation(records, args.size)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "double_annotation_queue.jsonl"
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected), encoding="utf-8")
    summary = {
        "experiment": "E012", "source_records": len(records), "selected_records": len(selected),
        "selection": "round-robin sorted primary_stratum then ascending ID",
        "ids": [record["id"] for record in selected],
        "operation_hint": dict(sorted(Counter(record["features"]["operation_hint"] for record in selected).items())),
        "primary_stratum": dict(sorted(Counter(record["features"]["primary_stratum"] for record in selected).items())),
        "output_path": str(output),
    }
    (args.output_dir / "double_annotation_manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
