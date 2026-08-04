#!/usr/bin/env python3
"""Final E011 tests: a repairable and a genuinely invalid byte-round-trip case."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifinqa.text_repair import repair_text


def main() -> None:
    repaired, adopted, codec = repair_text("Tá»•ng tÃ i sáº£n Ä‘áº¿n ngÃ y 31/12")
    assert adopted and codec == "cp1252" and repaired == "Tổng tài sản đến ngày 31/12", repaired
    clean = "Tổng tài sản đến ngày 31/12"
    assert repair_text(clean) == (clean, False, None)
    raw = "Ãÿ"  # CP1252/L1 bytes C3 FF cannot decode as UTF-8.
    assert repair_text(raw)[0] == raw
    print("E011 v3 text repair tests passed")


if __name__ == "__main__":
    main()
