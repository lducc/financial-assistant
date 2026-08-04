"""E006b corrected Vietnamese unit normalization."""

from __future__ import annotations

from decimal import Decimal
import re
import unicodedata

from .numeric import UnitInfo


NON_ALNUM = re.compile(r"[^a-z0-9%]+")


def normalized_text(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text.lower())
    deaccented = "".join(char for char in decomposed if unicodedata.category(char) != "Mn")
    return NON_ALNUM.sub(" ", deaccented.replace("đ", "d")).strip()


def detect_unit(text: str | None) -> UnitInfo:
    raw = "" if text is None else str(text)
    normalized = normalized_text(raw)
    if "%" in raw or "phan tram" in normalized or "ty le" in normalized:
        return UnitInfo(raw, "percent", None)
    patterns = (
        ("nghin ty", Decimal("1000000000000")),
        ("tram ty", Decimal("100000000000")),
        ("trieu dong", Decimal("1000000")),
        ("nghin dong", Decimal("1000")),
        ("ty dong", Decimal("1000000000")),
        ("vnd", Decimal("1")),
        ("dong", Decimal("1")),
    )
    for token, scale in patterns:
        if token in normalized:
            return UnitInfo(raw, "money", scale)
    return UnitInfo(raw, "unknown", None)

