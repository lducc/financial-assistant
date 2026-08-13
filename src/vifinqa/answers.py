"""Conservative numeric extraction for retrieved OCR evidence."""

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
import re
import unicodedata


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


# Cells are raw VND; 86.3% of questions ask for a scaled unit, and the answer
# tolerance is 0.02%, so an unconverted answer is wrong by six to nine orders of
# magnitude. Longest phrases first: "nghìn tỷ" must win over "tỷ".
OUTPUT_UNITS = (
    ("nghin ty dong", 1e12), ("nghin ty", 1e12),
    ("tram ty dong", 1e11), ("tram ty", 1e11),
    ("ty dong", 1e9), ("ty vnd", 1e9),
    ("trieu dong", 1e6), ("trieu vnd", 1e6),
    ("nghin dong", 1e3), ("nghin vnd", 1e3),
)
PERCENT_CUES = ("phan tram", "%", "diem phan tram")


def requested_scale(question: str) -> float:
    """The divisor that converts a raw VND figure into the unit the question asks for."""
    text = fold(question)
    for phrase, scale in OUTPUT_UNITS:
        if phrase in text:
            return scale
    return 1.0


def asks_percentage(question: str) -> bool:
    text = fold(question)
    return any(cue in text for cue in PERCENT_CUES)


def present(value: float, question: str, *, already_relative: bool = False) -> float:
    """Scale a computed figure into the requested unit and round as the spec asks.

    The organizers round only the final result, to two decimals, and return
    percentages as percentage points. Ratios and growth rates are already
    relative, so they are never divided by a currency scale.
    """
    if not already_relative:
        value = value / requested_scale(question)
    return round(value, 2)


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
    """Read a bound cell, in the row space the evaluator's DataFrame will have.

    Our row indices count the parsed grid, whose first row is the table's header.
    `pandas.read_csv` consumes that line as column names, so the same figure sits
    one row earlier in the DataFrame. Emitting the grid index made every query
    read the row below the intended one — which is why answer accuracy rose to
    0.1186 while execution accuracy stayed at 0.004.
    """
    expression = f"str({value.variable}.iloc[{max(0, value.row - 1)}, {value.column}])"
    normalized = f"{expression}.replace('(', '-').replace(')', '').replace('%', '').replace('.', '').replace(',', '.')"
    return f"float({normalized})"


def select_cell(
    rows: list[list[str]], question_tokens: set[str], headers: list[str] | None = None,
) -> tuple[int, int, float] | None:
    """Find the row whose label best matches the question, then read its value.

    Retrieval binds a row while ranking a whole table, so in principle the bound
    row is the one that made the table look relevant rather than the one asked
    for. Measured against the benchmark's 613 verified bindings, searching every
    row scores identically to trusting the retrieval binding — 202 exact cells
    either way, with five row errors traded for five column errors. Kept as the
    instrument that established that, and used by scripts/evaluate_cells.py;
    production still reads the retrieval binding.

    Ties go to the earlier row, so statements keep their natural order.
    """
    best: tuple[float, int, int, float] | None = None
    for index, row in enumerate(rows):
        if not row:
            continue
        label = " ".join(fold(cell) for cell in row[:2])
        overlap = sum(1 for token in question_tokens if token in label)
        if not overlap:
            continue
        numeric = first_numeric_cell(list(row), headers)
        if numeric is None:
            continue
        column, value = numeric
        score = overlap / max(1, len(question_tokens))
        if best is None or score > best[0]:
            best = (score, index, column, value)
    return (best[1], best[2], best[3]) if best else None


def says(text: str, *phrases: str) -> bool:
    """Whole-word phrase match.

    Folding removes the diacritics that separate "hiệu" (difference) from "nhiêu"
    in "bao nhiêu", which ends nearly every question, so a substring test routes
    almost everything to subtraction.
    """
    return any(re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text) for phrase in phrases)


def operator_text(question: str) -> str:
    """The question with its line-item names removed, leaving the arithmetic wording.

    "Tổng phải thu ngắn hạn khác của OGC" is a lookup, not a sum: "tổng" belongs to
    the line item. Matching operator keywords against the raw question routed 106
    such lookups into sums. Stripping the corpus's own row labels first leaves only
    the words that describe an operation.
    """
    text = f" {fold(question)} "
    for label in line_item_phrases():
        if label in text:
            text = text.replace(label, " ")
    return text


@lru_cache(maxsize=1)
def line_item_phrases() -> tuple[str, ...]:
    """Corpus row labels, longest first, from scripts/build_line_item_lexicon.py."""
    path = Path(__file__).resolve().parents[2] / "data" / "derived" / "line_items.json"
    if not path.exists():
        return ()
    labels = json.loads(path.read_text(encoding="utf-8"))
    return tuple(sorted(labels, key=lambda label: (-len(label), label)))


def answer_plan(question: str, values: list[EvidenceValue]) -> tuple[float, str] | None:
    """Plan only common arithmetic when every operand is source-bound evidence.

    The returned figure is in the unit the question asks for and rounded to two
    decimals, because the scorer compares against that within 0.02%.
    """
    if not values:
        return None
    text = operator_text(question)
    expressions = [numeric_expression(value) for value in values]
    scale = requested_scale(question)
    if len(values) == 1:
        return present(values[0].value, question), f"({expressions[0]} / {scale})"
    if says(text, "tang truong", "toc do tang"):
        if values[0].value == 0:
            return None
        growth = (values[-1].value - values[0].value) / values[0].value * 100
        return present(growth, question, already_relative=True), (
            f"(({expressions[-1]} - {expressions[0]}) / {expressions[0]} * 100)"
        )
    if says(text, "trung binh", "binh quan"):
        mean = sum(value.value for value in values) / len(values)
        return present(mean, question), f"((({' + '.join(expressions)}) / {len(values)}) / {scale})"
    if says(text, "tong", "cong", "tong cong"):
        return present(sum(value.value for value in values), question), (
            f"(({' + '.join(expressions)}) / {scale})"
        )
    if says(text, "chenh lech", "hieu", "tru", "cao hon", "thap hon", "lon hon"):
        difference = values[0].value - sum(value.value for value in values[1:])
        return present(difference, question), (
            f"(({expressions[0]} - ({' + '.join(expressions[1:])})) / {scale})"
        )
    if says(text, "ty le", "ty trong", "gap", "bien loi nhuan", "tren"):
        if values[1].value == 0:
            return None
        multiplier = 100 if asks_percentage(question) else 1
        ratio = values[0].value / values[1].value * multiplier
        return present(ratio, question, already_relative=True), (
            f"({expressions[0]} / {expressions[1]} * {multiplier})"
        )
    if says(text, "lon nhat", "cao nhat", "nhieu nhat"):
        return present(max(value.value for value in values), question), (
            f"(max({', '.join(expressions)}) / {scale})"
        )
    if says(text, "nho nhat", "thap nhat", "it nhat"):
        return present(min(value.value for value in values), question), (
            f"(min({', '.join(expressions)}) / {scale})"
        )
    return None
