#!/usr/bin/env python3
"""E006b regression fixtures using corrected Vietnamese unit detection."""

from decimal import Decimal
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifinqa.numeric import parse_number, to_base_vnd
from vifinqa.numeric_v2 import detect_unit


def main() -> None:
    values = {
        "1.234.567": Decimal("1234567"),
        "1.234,56": Decimal("1234.56"),
        "1,234.56": Decimal("1234.56"),
        "(822.663.402)": Decimal("-822663402"),
        "100,00%": Decimal("100"),
        "24.96%": Decimal("24.96"),
    }
    for raw, expected in values.items():
        assert parse_number(raw).value == expected
    assert parse_number("-").status == "missing"
    assert parse_number("12,3456").status == "ambiguous_separator"
    assert detect_unit("Đơn vị: triệu đồng").scale_to_vnd == Decimal("1000000")
    assert detect_unit("Nghìn tỷ đồng").scale_to_vnd == Decimal("1000000000000")
    assert detect_unit("Tỷ lệ (%)").kind == "percent"
    assert to_base_vnd(parse_number("1,25"), detect_unit("triệu đồng")) == Decimal("1250000.00")
    print("E006b fixtures passed")


if __name__ == "__main__":
    main()

