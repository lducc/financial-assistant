#!/usr/bin/env python3
"""E014 unit tests."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifinqa.heuristic_audit import compare_operation_hints


def main() -> None:
    queue = [{"id": 1, "features": {"operation_hint": "extremum_or_selector"}}, {"id": 2, "features": {"operation_hint": "lookup"}}]
    plans = [{"id": 1, "operation": "selector"}, {"id": 2, "operation": "difference"}]
    result = compare_operation_hints(queue, plans)
    assert result["agree_count"] == 1 and result["disagree_count"] == 1
    assert result["disagreements"][0]["id"] == 2
    print("E014 heuristic audit tests passed")


if __name__ == "__main__":
    main()
