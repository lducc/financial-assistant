"""Conservative numeric extraction for retrieved OCR evidence."""

import re
import unicodedata
from dataclasses import dataclass


NUMBER_RE = re.compile(r"[+-]?\d[\d.,]*")
GROUPED_RE = re.compile(r"\d{1,3}([.,]\d{3})+(?![\d])")
SKIP_HEADERS = ("ma so", "thuyet minh", "stt", "ghi chu")


def parse_ocr_number(value: str) -> float | None:
    """Parse a Vietnamese-formatted numeric OCR cell without guessing text values."""
    match = NUMBER_RE.search(value.replace("(", "-").replace(")", ""))
    if not match:
        return None
    number = match.group(0)
    if "," in number and "." in number:
        number = number.replace(".", "").replace(",", ".")
    elif number.count(".") > 1 or ("." in number and len(number.rsplit(".", 1)[1]) == 3):
        number = number.replace(".", "")
    elif number.count(",") > 1 or ("," in number and len(number.rsplit(",", 1)[1]) == 3):
        number = number.replace(",", "")
    else:
        number = number.replace(",", ".")
    try:
        return float(number)
    except ValueError:
        return None


def fold(text: str) -> str:
    """Strip Vietnamese diacritics and casefold, matching retrieval-side folding."""
    text = unicodedata.normalize("NFD", text.casefold())
    return "".join(char for char in text if unicodedata.category(char) != "Mn").replace("đ", "d")


def first_numeric_cell(cells: list[str], headers: list[str] | None = None) -> tuple[int, float] | None:
    """Return the value cell after a row label, skipping code/note columns."""
    tiers: list[list[tuple[int, float]]] = [[], [], [], []]
    for column, cell in enumerate(cells[1:], 1):
        if headers and column < len(headers):
            folded = fold(headers[column])
            if any(marker in folded for marker in SKIP_HEADERS):
                continue
        value = parse_ocr_number(cell)
        if value is None:
            continue
        if GROUPED_RE.search(cell):
            tiers[0].append((column, value))
        elif abs(value) >= 1000:
            tiers[1].append((column, value))
        elif "%" in cell or value != int(value):
            tiers[2].append((column, value))
        else:
            tiers[3].append((column, value))
    for tier in tiers:
        if tier:
            return tier[0]
    return None


def cell_expression(variable: str, row: int, column: int) -> str:
    """Create a pandas-compatible expression matching common Vietnamese number formatting."""
    value = f"str({variable}.iloc[{row}, {column}])"
    normalized = f"{value}.replace('(', '-').replace(')', '').replace('%', '').replace('.', '').replace(',', '.')"
    return f"result = float({normalized})"


@dataclass(frozen=True)
class EvidenceValue:
    variable: str
    row: int
    column: int
    value: float
    report_id: str


def numeric_expression(value: EvidenceValue) -> str:
    expression = f"str({value.variable}.iloc[{value.row}, {value.column}])"
    normalized = f"{expression}.replace('(', '-').replace(')', '').replace('%', '').replace('.', '').replace(',', '.')"
    return f"float({normalized})"


def answer_plan(question: str, values: list[EvidenceValue]) -> tuple[float, str] | None:
    """Plan only common arithmetic when every operand is source-bound evidence."""
    if not values:
        return None
    text = question.lower()
    expressions = [numeric_expression(value) for value in values]
    if len(values) == 1:
        return values[0].value, expressions[0]
    if "tăng trưởng" in text or "tốc độ tăng" in text:
        if values[0].value == 0:
            return None
        return (values[-1].value - values[0].value) / values[0].value * 100, f"(({expressions[-1]} - {expressions[0]}) / {expressions[0]} * 100)"
    if "trung bình" in text or "bình quân" in text:
        return sum(value.value for value in values) / len(values), f"(({ ' + '.join(expressions) }) / {len(values)})"
    if "tổng" in text or "cộng" in text:
        return sum(value.value for value in values), " + ".join(expressions)
    if "chênh lệch" in text or "hiệu" in text or "trừ" in text:
        return values[0].value - sum(value.value for value in values[1:]), f"({expressions[0]} - ({' + '.join(expressions[1:])}))"
    if "tỷ lệ" in text or "tỷ trọng" in text or " gấp " in f" {text} ":
        if values[1].value == 0:
            return None
        multiplier = 100 if ("%" in text or "phần trăm" in text) else 1
        return values[0].value / values[1].value * multiplier, f"({expressions[0]} / {expressions[1]} * {multiplier})"
    if "lớn nhất" in text or "cao nhất" in text:
        return max(value.value for value in values), f"max({', '.join(expressions)})"
    if "nhỏ nhất" in text or "thấp nhất" in text:
        return min(value.value for value in values), f"min({', '.join(expressions)})"
    return None
