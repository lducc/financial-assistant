#!/usr/bin/env python3
"""Focused E009 planner guards; no dataset labels are assumed."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifinqa.planner import build_plan, classify_operation


def expect(question: str, operation: str) -> None:
    actual = classify_operation(question).operation
    assert actual == operation, (question, actual, operation)


def main() -> None:
    expect("Doanh thu tăng trưởng năm 2023 là bao nhiêu?", "growth_or_change")
    expect("Tỷ lệ nợ trên vốn chủ sở hữu là bao nhiêu phần trăm?", "ratio_or_percent")
    expect("Giá trị trung bình trong ba năm là bao nhiêu?", "average")
    expect("Năm nào có doanh thu cao nhất?", "selector")
    expect("Chênh lệch lợi nhuận giữa hai năm là bao nhiêu?", "difference")
    expect("Tổng cộng các khoản chi phí là bao nhiêu?", "aggregate")
    expect("Tổng tài sản năm 2023 là bao nhiêu?", "lookup")
    plan = build_plan("Tỷ lệ cổ tức là bao nhiêu %?", {"tickers": ["VNM"], "years": [2023], "scope": "consolidated"})
    assert plan["output_unit_hint"] == "percent_points"
    assert plan["constraints"]["tickers"] == ["VNM"]
    print("E009 planner tests passed")


if __name__ == "__main__":
    main()
