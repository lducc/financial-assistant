"""Transparent parsing of Vietnamese financial number and unit forms."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import re
import unicodedata


MISSING = {"", "-", "--", "–", "—", "n/a", "na"}
NON_ALNUM = re.compile(r"[^a-z0-9%]+")


@dataclass(frozen=True)
class ParsedNumber:
    raw: str
    value: Decimal | None
    is_percent: bool
    status: str


@dataclass(frozen=True)
class UnitInfo:
    raw: str
    kind: str
    scale_to_vnd: Decimal | None


def normalized_text(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text.lower())
    deaccented = "".join(char for char in decomposed if unicodedata.category(char) != "Mn")
    return NON_ALNUM.sub(" ", deaccented).strip()


def parse_number(raw: str | None) -> ParsedNumber:
    source = "" if raw is None else str(raw)
    text = source.replace("\u00a0", " ").strip()
    if text.lower() in MISSING:
        return ParsedNumber(source, None, False, "missing")
    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1].strip()
    if text.startswith(("-", "−")):
        negative = True
        text = text[1:].strip()
    is_percent = "%" in text
    text = text.replace("%", "").replace(" ", "")
    if not text or re.search(r"[^0-9,\.]+", text):
        return ParsedNumber(source, None, is_percent, "unparsed")
    comma_count, dot_count = text.count(","), text.count(".")
    if comma_count and dot_count:
        decimal_separator = "," if text.rfind(",") > text.rfind(".") else "."
        thousand_separator = "." if decimal_separator == "," else ","
        normalized = text.replace(thousand_separator, "").replace(decimal_separator, ".")
    elif comma_count or dot_count:
        separator = "," if comma_count else "."
        count = comma_count or dot_count
        before, after = text.rsplit(separator, 1)
        if count > 1 or len(after) == 3:
            normalized = text.replace(separator, "")
        elif len(after) in {1, 2}:
            normalized = before.replace(separator, "") + "." + after
        else:
            return ParsedNumber(source, None, is_percent, "ambiguous_separator")
    else:
        normalized = text
    try:
        value = Decimal(normalized)
    except InvalidOperation:
        return ParsedNumber(source, None, is_percent, "unparsed")
    if negative:
        value = -value
    return ParsedNumber(source, value, is_percent, "ok")


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


def to_base_vnd(number: ParsedNumber, unit: UnitInfo) -> Decimal | None:
    if number.value is None or unit.kind != "money" or unit.scale_to_vnd is None:
        return None
    return number.value * unit.scale_to_vnd

