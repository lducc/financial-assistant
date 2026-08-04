#!/usr/bin/env python3
"""Curated E006 fixtures for numeric and unit parsing."""

from decimal import Decimal
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifinqa.numeric import detect_unit, parse_number, to_base_vnd


def check(raw: str, expected: str | None, status: str = "ok") -> None:
    parsed = parse_number(raw)
    assert parsed.status == status, (raw, parsed)
    assert parsed.value == (Decimal(expected) if expected is not None else None), (raw, parsed)


def main() -> None:
    check("1.234.567", "1234567")
    check("1.234,56", "1234.56")
    check("1,234.56", "1234.56")
    check("(822.663.402)", "-822663402")
    check("100,00%", "100")
    check("24.96%", "24.96")
    check("-", None, "missing")
    check("N/A", None, "missing")
    check("12,3456", None, "ambiguous_separator")
    check("abc", None, "unparsed")
    assert detect_unit("Đơn vị: triệu đồng").scale_to_vnd == Decimal("1000000")
    assert detect_unit("Nghìn tỷ đồng").scale_to_vnd == Decimal("1000000000000")
    assert detect_unit("Tỷ lệ (%)").kind == "percent"
    assert to_base_vnd(parse_number("1,25"), detect_unit("triệu đồng")) == Decimal("1250000.00")
    print("E006 fixtures passed")


if __name__ == "__main__":
    main()

