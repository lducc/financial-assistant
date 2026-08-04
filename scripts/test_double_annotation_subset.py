#!/usr/bin/env python3
"""E012 reproducibility and coverage tests on the frozen local queue."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifinqa.double_annotation import select_double_annotation


def main() -> None:
    queue = ROOT / "data" / "derived" / "proxy_queue" / "annotation_queue.jsonl"
    records = [json.loads(line) for line in queue.read_text(encoding="utf-8").splitlines() if line.strip()]
    first = select_double_annotation(records)
    second = select_double_annotation(records)
    assert [row["id"] for row in first] == [row["id"] for row in second]
    assert len(first) == len({row["id"] for row in first}) == 30
    source_operations = {row["features"]["operation_hint"] for row in records}
    selected_operations = {row["features"]["operation_hint"] for row in first}
    assert source_operations <= selected_operations, (source_operations, selected_operations)
    print("E012 double-annotation subset tests passed")


if __name__ == "__main__":
    main()
